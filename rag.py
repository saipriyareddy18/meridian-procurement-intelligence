"""
Retrieve + answer — tuned for assignment accuracy.
Campus-approved Gemini (or OpenAI / Ollama via LLM_PROVIDER).
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from config import (
    chat_model,
    ensure_provider_ready,
    get_gemini_api_key,
    openai_client_kwargs,
    provider,
)
from dotenv import load_dotenv
from langchain_core.documents import Document

from ingest import COLLECTION_NAME, _cache_key, chunk_count_fast, embedding_model, get_vectorstore

load_dotenv()

TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "900"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))

SYSTEM_PROMPT = """You are a procurement assistant for Meridian Components Pvt. Ltd.
Answer ONLY using the context below. Do not use outside knowledge.

Rules:
1. If the answer truly cannot be derived from the context, reply exactly:
   The information is not available in the uploaded documents.
2. If the context has the figures AND the rule, you MUST combine them and answer (do not refuse).
3. When a question needs a number/figure AND a policy rule, state clearly:
   - Figure (with units)
   - Clause / rule triggered
   - Required buyer action
4. For safety stock: compute lead_time × 0.25, compare with the policy minimum floor, and use the HIGHER value.
5. For "below B band on on-time delivery alone": use the handbook rule that OTD below 75% cannot score in band B; list matching suppliers from the scorecard (or say none if none match), then state the escalation path from the handbook.
6. Be exact with clause numbers, percentages, ₹ amounts, and supplier names.
7. Never invent salaries, penalties, approvals, or figures not present in the context.
"""

_ANSWER_CACHE: dict[str, dict[str, Any]] = {}
_ANSWER_CACHE_MAX = 64


def _qkey(question: str, top_k: int) -> str:
    raw = f"{provider()}|{chat_model()}|{top_k}|{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


@lru_cache(maxsize=4)
def _chat_llm_cached(cache_key: str, model: str):
    p = provider()
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_gemini_api_key(),
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=TEMPERATURE, num_predict=MAX_OUTPUT_TOKENS)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        **openai_client_kwargs(),
    )


def _chat_llm():
    return _chat_llm_cached(_cache_key(), chat_model())


def _search(store, question: str, k: int, doc_type: str | None = None) -> list[Document]:
    try:
        if doc_type:
            return store.similarity_search(question, k=k, filter={"doc_type": doc_type})
        return store.similarity_search(question, k=k)
    except Exception:
        return []


def retrieve_balanced(question: str, top_k: int = 6, per_doc: int | None = None) -> list[Document]:
    """Always pull from review + policy in parallel (cross-document accuracy)."""
    if chunk_count_fast() == 0:
        return []

    store = get_vectorstore()
    if per_doc is None:
        per_doc = max(3, (top_k + 1) // 2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_r = pool.submit(_search, store, question, per_doc, "review")
        fut_p = pool.submit(_search, store, question, per_doc, "policy")
        review = fut_r.result()
        policy = fut_p.result()

    if not review and not policy:
        return _search(store, question, top_k)

    merged = review + policy
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in merged:
        key = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content[:50]}"
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique[: max(top_k, len(unique))]


def format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        text = doc.page_content
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "…"
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("doc_type", "other")
        parts.append(f"[Chunk {i} | file={src} | page={page} | type={dtype}]\n{text}")
    return "\n\n".join(parts)


def sources_from_docs(docs: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for doc in docs:
        file_name = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (file_name, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file": file_name,
                "page": page,
                "doc_type": doc.metadata.get("doc_type", "other"),
            }
        )
    return sources


def group_sources_by_document(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        grouped.setdefault(s["file"], []).append(s)
    return grouped


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def ask(question: str, top_k: int = 6) -> dict[str, Any]:
    ensure_provider_ready()
    key = _qkey(question, top_k)
    cached = _ANSWER_CACHE.get(key)
    if cached:
        out = dict(cached)
        out["cached"] = True
        return out

    t0 = time.perf_counter()
    docs = retrieve_balanced(question, top_k=top_k)
    if not docs:
        return {
            "answer": "No documents are indexed yet. Click Index Documents in the sidebar first.",
            "sources": [],
            "grouped_sources": {},
            "retrieved_count": 0,
            "cached": False,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    context = format_context(docs)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context. Be precise. For safety stock use higher of formula vs floor."
    )

    llm = _chat_llm()
    last_err: Exception | None = None
    response = None
    for attempt in range(4):
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if "503" in msg or "unavailable" in msg or "high demand" in msg:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if response is None:
        raise RuntimeError(f"Model unavailable after retries: {last_err}")
    answer = _normalize_content(response.content)
    sources = sources_from_docs(docs)

    result = {
        "answer": answer,
        "sources": sources,
        "grouped_sources": group_sources_by_document(sources),
        "retrieved_count": len(docs),
        "cached": False,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "models": {
            "embedding": embedding_model(),
            "chat": chat_model(),
            "provider": provider(),
        },
        "collection": COLLECTION_NAME,
    }

    if len(_ANSWER_CACHE) >= _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))
    _ANSWER_CACHE[key] = {k: v for k, v in result.items() if k != "cached"}
    return result


def clear_answer_cache() -> None:
    _ANSWER_CACHE.clear()


def smoke_test_api() -> str:
    ensure_provider_ready()
    llm = _chat_llm()
    response = llm.invoke("Reply with exactly: OK")
    return _normalize_content(response.content).strip()
