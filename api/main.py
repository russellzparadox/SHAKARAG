from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import make_pipeline

app = FastAPI(title="Odoo Schema RAG", version="0.1.0")
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = make_pipeline(load_settings_safe())
    return _pipeline


def load_settings_safe():
    from rag.config import load_settings

    return load_settings()


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    execute: bool = True
    dry_run: bool = False
    top_k: Optional[int] = None


@app.get("/health")
def health() -> dict[str, Any]:
    settings = load_settings_safe()
    try:
        pipeline = get_pipeline()
        indexed = pipeline.store.count
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"index unavailable: {exc}")
    return {
        "status": "ok",
        "indexed_vectors": indexed,
        "llm_ready": settings.llm_ready,
        "dialect": pipeline.dialect.label,
        "db": f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    pipeline = get_pipeline()
    result = pipeline.ask(
        req.question,
        execute=req.execute,
        dry_run=req.dry_run,
        top_k=req.top_k,
    )
    return {
        "question": result.question,
        "tables_used": result.tables_used,
        "sql": result.sql,
        "explanation": result.explanation,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "answer": result.answer,
        "error": result.error,
    }


class ReindexRequest(BaseModel):
    drop: bool = True


@app.post("/reindex")
def reindex(req: ReindexRequest) -> dict[str, Any]:
    global _pipeline
    from scripts.ingest import run_ingest

    stats = run_ingest(drop=req.drop)
    _pipeline = None
    return stats


@app.get("/debug/context")
def debug_context(q: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    hits = pipeline.retrieve(q)
    context, tables = pipeline.build_context(hits)
    return {"tables": tables, "context": context[:4000]}
