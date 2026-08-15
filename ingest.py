"""
Load PDFs, chunk text, embed, store in ChromaDB.
Cached clients + skip re-index when PDFs unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    chat_model,
    embedding_model,
    ensure_provider_ready,
    get_gemini_api_key,
    openai_client_kwargs,
    provider,
)
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
META_PATH = CHROMA_DIR / "ingest_meta.json"
COLLECTION_NAME = "meridian_supply_chain"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Gemini free tier: ~100 embed requests/min. Small batches + retries avoid 429s.
EMBED_BATCH_SIZE = 8
EMBED_BATCH_PAUSE_SEC = 1.2
EMBED_MAX_RETRIES = 8

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).upper()
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "RATE LIMIT" in text
        or "QUOTA" in text
    )


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    """Prefer API retryDelay when present; otherwise exponential backoff."""
    text = str(exc)
    match = re.search(r"retry(?:Delay| in)\D*([0-9]+(?:\.[0-9]+)?)\s*s", text, re.I)
    if match:
        return max(float(match.group(1)) + 1.0, 5.0)
    return min(5.0 * (2**attempt), 90.0)


def _add_documents_with_retry(
    store: Chroma,
    chunks: list[Document],
    ids: list[str],
) -> None:
    """Embed in small batches with backoff so free-tier Gemini does not 429."""
    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        batch_docs = chunks[start:end]
        batch_ids = ids[start:end]
        last_exc: BaseException | None = None
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                store.add_documents(documents=batch_docs, ids=batch_ids)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 — surface after retries
                last_exc = exc
                if not _is_rate_limit_error(exc) or attempt >= EMBED_MAX_RETRIES - 1:
                    raise
                delay = _retry_delay_seconds(exc, attempt)
                print(
                    f"Embed rate limit on chunks {start + 1}-{end}/{total}; "
                    f"waiting {delay:.0f}s (attempt {attempt + 1}/{EMBED_MAX_RETRIES})..."
                )
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        if end < total:
            time.sleep(EMBED_BATCH_PAUSE_SEC)


def _cache_key() -> str:
    """Key so embeddings/Chroma clients are reused across asks."""
    p = provider()
    model = embedding_model()
    if p == "gemini":
        secret = get_gemini_api_key()[:12]
    elif p == "openai":
        secret = openai_client_kwargs()["api_key"][:12]
    else:
        secret = "ollama"
    return f"{p}|{model}|{secret}"


@lru_cache(maxsize=4)
def get_embeddings_cached(cache_key: str):
    p = provider()
    model = embedding_model()
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=get_gemini_api_key(),
        )
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, **openai_client_kwargs())


def get_embeddings():
    return get_embeddings_cached(_cache_key())


def reset_caches() -> None:
    get_embeddings_cached.cache_clear()
    get_vectorstore_cached.cache_clear()


@lru_cache(maxsize=4)
def get_vectorstore_cached(cache_key: str) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings_cached(cache_key),
        persist_directory=str(CHROMA_DIR),
    )


def get_vectorstore(create_if_missing: bool = True) -> Chroma:
    return get_vectorstore_cached(_cache_key())


def detect_document_type(filename: str) -> str:
    name = filename.lower()
    if "handbook" in name or "policy" in name or "procurement" in name:
        return "policy"
    if "review" in name or "scorecard" in name or "supply_chain" in name:
        return "review"
    return "other"


def extract_pages(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    docs: list[Document] = []
    doc_type = detect_document_type(pdf_path.name)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": i + 1,
                    "doc_type": doc_type,
                },
            )
        )
    return docs


def chunk_documents(pages: list[Document]) -> list[Document]:
    return _SPLITTER.split_documents(pages)


def _stable_id(doc: Document, index: int) -> str:
    raw = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{index}|{doc.page_content[:80]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pdf_fingerprint(pdf_paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(pdf_paths, key=lambda p: p.name):
        st = path.stat()
        h.update(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{provider()}|{embedding_model()}".encode())
    return h.hexdigest()


def _read_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(payload: dict[str, Any]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(payload), encoding="utf-8")


def chunk_count_fast() -> int:
    """Count without rebuilding embedding clients when possible."""
    meta = _read_meta()
    if "chunks" in meta:
        return int(meta["chunks"])
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return 0
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def clear_collection() -> None:
    reset_caches()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def ingest_pdfs(
    pdf_paths: list[Path],
    clear_first: bool = True,
    skip_if_unchanged: bool = True,
) -> dict[str, Any]:
    if not pdf_paths:
        raise ValueError("No PDF paths provided.")

    ensure_provider_ready()
    paths = [Path(p) for p in pdf_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

    fingerprint = _pdf_fingerprint(paths)
    meta = _read_meta()
    existing = chunk_count_fast()

    if (
        skip_if_unchanged
        and existing > 0
        and meta.get("fingerprint") == fingerprint
    ):
        return {
            "files": len(paths),
            "chunks": existing,
            "skipped": True,
            "message": "Already indexed — skipped rebuild",
            "collection": COLLECTION_NAME,
            "embedding_model": embedding_model(),
            "provider": provider(),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

    # Avoid duplicate chunks when PDFs/provider changed
    if clear_first or existing > 0:
        clear_collection()

    all_pages: list[Document] = []
    per_file_pages: dict[str, int] = {}
    for path in paths:
        pages = extract_pages(path)
        per_file_pages[path.name] = len(pages)
        all_pages.extend(pages)

    chunks = chunk_documents(all_pages)
    per_file_chunks: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        per_file_chunks[src] = per_file_chunks.get(src, 0) + 1

    ids = [_stable_id(c, i) for i, c in enumerate(chunks)]
    reset_caches()
    store = get_vectorstore()
    _add_documents_with_retry(store, chunks, ids)

    result = {
        "files": len(paths),
        "chunks": len(chunks),
        "skipped": False,
        "pages_per_file": per_file_pages,
        "chunks_per_file": per_file_chunks,
        "collection": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": embedding_model(),
        "provider": provider(),
    }
    _write_meta(
        {
            "fingerprint": fingerprint,
            "chunks": len(chunks),
            "files": [p.name for p in paths],
            "provider": provider(),
            "embedding_model": embedding_model(),
        }
    )
    return result


def ingest_data_folder(clear_first: bool = True, skip_if_unchanged: bool = True) -> dict[str, Any]:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")
    return ingest_pdfs(pdfs, clear_first=clear_first, skip_if_unchanged=skip_if_unchanged)


def collection_stats() -> dict[str, Any]:
    count = chunk_count_fast()
    return {
        "collection": COLLECTION_NAME,
        "chunks": count,
        "persist_directory": str(CHROMA_DIR),
        "embedding_model": embedding_model(),
        "chat_model": chat_model(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "provider": provider(),
    }


# Back-compat alias used by older docs
EMBEDDING_MODEL = embedding_model()


if __name__ == "__main__":
    print("Extracting and indexing PDFs from data/ ...")
    result = ingest_data_folder(skip_if_unchanged=False)
    print(result)
    print("Stats after ingest:", collection_stats())
