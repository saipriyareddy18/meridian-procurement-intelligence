"""
Optional FastAPI backend — bonus stage.
Endpoints: POST /ingest, POST /ask, GET /stats
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ingest import collection_stats, ingest_pdfs
from rag import ask

app = FastAPI(
    title="Meridian Supply Chain RAG API",
    description="Ingest PDFs, ask grounded questions, inspect Chroma stats.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=12)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Meridian Supply Chain RAG API",
        "docs": "/docs",
        "endpoints": "/ingest, /ask, /stats",
    }


@app.post("/ingest")
async def ingest_endpoint(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sc_api_"))
    paths: list[Path] = []
    try:
        for f in files:
            if not f.filename or not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
            dest = tmp_dir / Path(f.filename).name
            dest.write_bytes(await f.read())
            paths.append(dest)

        result = ingest_pdfs(paths, clear_first=True, skip_if_unchanged=False)
        return {"files": result["files"], "chunks": result["chunks"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    try:
        result = ask(body.question, top_k=body.top_k)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def stats_endpoint() -> dict[str, Any]:
    try:
        return collection_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
