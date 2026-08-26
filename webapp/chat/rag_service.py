from __future__ import annotations

import logging
import threading
from typing import Any

from rag.config import Settings as RagSettings
from rag.config import load_settings as _load_base
from rag.pipeline import RagPipeline, make_pipeline as _make_pipeline

from .crypto import decrypt

_base_settings = None


def base_settings() -> RagSettings:
    global _base_settings
    if _base_settings is None:
        _base_settings = _load_base()
    return _base_settings


def build_rag_settings(dbp, llmp) -> RagSettings:
    base = base_settings()
    return RagSettings(
        db_dialect=dbp.dialect,
        db_host=dbp.host,
        db_port=dbp.port or 0,
        db_user=dbp.db_user,
        db_password=decrypt(dbp.password_enc),
        db_name=dbp.db_name,
        db_url=dbp.db_url or None,
        chroma_dir=base.chroma_dir,
        collection=dbp.collection_name,
        embed_provider=base.embed_provider,
        openai_api_key=base.openai_api_key,
        openai_base_url=base.openai_base_url,
        embed_model=base.embed_model,
        llm_base_url=llmp.base_url if llmp else None,
        llm_model=llmp.model if llmp else None,
        llm_api_key=decrypt(llmp.api_key_enc) if llmp else None,
        llm_temperature=llmp.temperature if llmp else base.llm_temperature,
        top_k=base.top_k,
        context_char_budget=base.context_char_budget,
        max_rows=base.max_rows,
        statement_timeout_ms=base.statement_timeout_ms,
        sample_values=base.sample_values,
        value_sample_max_rows=base.value_sample_max_rows,
        examples_top_k=base.examples_top_k,
        data_preview=base.data_preview,
    )


_pipeline_cache: dict[tuple, RagPipeline] = {}


def cache_key(dbp, llmp) -> tuple:
    return (
        dbp.pk,
        dbp.collection_name,
        int(dbp.updated_at.timestamp() * 1000),
        llmp.pk if llmp else None,
        int(llmp.updated_at.timestamp() * 1000) if llmp else 0,
    )


def get_pipeline(dbp, llmp) -> RagPipeline:
    key = cache_key(dbp, llmp)
    pipeline = _pipeline_cache.get(key)
    if pipeline is None:
        settings = build_rag_settings(dbp, llmp)
        pipeline = _make_pipeline(settings)
        _pipeline_cache.clear()
        _pipeline_cache[key] = pipeline
    return pipeline


def run_ask(
    dbp,
    llmp,
    question: str,
    answer_language: str = "auto",
    allow_clarify: bool = False,
    history: list[dict] | None = None,
) -> dict:
    logger = logging.getLogger("chat.ask")
    try:
        pipeline = get_pipeline(dbp, llmp)
        result = pipeline.ask(
            question,
            execute=True,
            answer_language=answer_language,
            clarify=allow_clarify,
            history=history,
        )
        logger.info(
            "ask ok db=%s llm=%s clarify=%s tables=%s rows=%s",
            dbp.name,
            llmp.name if llmp else "-",
            result.needs_clarification,
            result.tables_used,
            result.row_count,
        )
    except Exception as exc:
        logger.exception("ask failed db=%s llm=%s", dbp.name, llmp.name if llmp else "-")
        return {"error": str(exc), "answer": f"Sorry — the request failed: {exc}"}
    answer = result.answer
    if not answer and result.sql and not result.error:
        answer = "SQL generated (not answered — check LLM config):\n" + result.sql
    elif not answer and result.error:
        answer = f"Request failed: {result.error}"
    return {
        "clarify": result.needs_clarification,
        "clarify_question": result.clarify_question or "",
        "options": result.options,
        "route": result.route,
        "doc_sources": result.doc_sources,
        "sql": result.sql,
        "explanation": result.explanation,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "tables_used": result.tables_used,
        "answer": answer,
        "error": result.error,
    }


def example_store(dbp):
    from rag.examples import ExampleStore
    from rag.embeddings import get_embedder

    key = (dbp.collection_name, int(dbp.updated_at.timestamp() * 1000))
    store = _example_cache.get(dbp.pk)
    if store is None or store._cache_key != key:
        settings = build_rag_settings(dbp, None)
        embedder = get_embedder(settings)
        store = ExampleStore(str(settings.chroma_dir), settings.collection, embedder)
        store._cache_key = key
        _example_cache[dbp.pk] = store
    return store


_example_cache: dict[int, Any] = {}
_schema_store_cache: dict[int, Any] = {}


def schema_store(dbp):
    """Vector store for the dbp's main collection (schema + documents)."""
    store = _schema_store_cache.get(dbp.pk)
    key = (dbp.collection_name, int(dbp.updated_at.timestamp() * 1000))
    if store is None or getattr(store, "_cache_key", None) != key:
        from rag.embeddings import get_embedder
        from rag.store import VectorStore

        settings = build_rag_settings(dbp, None)
        embedder = get_embedder(settings)
        store = VectorStore(str(settings.chroma_dir), settings.collection, embedder)
        store._cache_key = key
        _schema_store_cache[dbp.pk] = store
    return store


def ingest_document(dbp, filename: str, data: bytes) -> dict:
    """Extract + index a PDF/Word/Excel document into the dbp's vector collection."""
    from rag.documents import DocIngestError, ingest_document_bytes

    try:
        stats = ingest_document_bytes(schema_store(dbp), filename, data)
        logging.getLogger("chat.docs").info(
            "document indexed db=%s file=%s chunks=%s",
            dbp.name,
            filename,
            stats["chunks"],
        )
        return stats
    except DocIngestError as exc:
        raise ValueError(str(exc)) from exc


def record_feedback(dbp, question: str, sql: str, notes: str = "", helpful: bool = True) -> str:
    store = example_store(dbp)
    if helpful:
        ex_id = store.add(question=question, sql=sql, notes=notes)
        logging.getLogger("chat.feedback").info(
            "example stored for db=%s id=%s", dbp.name, ex_id
        )
        return "saved"
    removed = store.remove(question)
    logging.getLogger("chat.feedback").info(
        "example %s for db=%s", "removed" if removed else "absent", dbp.name
    )
    return "removed"


def start_reindex(dbp) -> None:
    thread = threading.Thread(target=_reindex_worker, args=(dbp.pk,), daemon=True)
    thread.start()


def _reindex_worker(dbp_id: int) -> None:
    import logging
    import time
    import traceback

    from django.db import close_old_connections
    from django.utils import timezone

    from chat.models import DatabaseProfile

    logger = logging.getLogger("chat.indexing")
    close_old_connections()
    try:
        dbp = DatabaseProfile.objects.get(pk=dbp_id)
        dbp.index_status = DatabaseProfile.IndexStatus.INDEXING
        dbp.index_error = ""
        dbp.save(update_fields=["index_status", "index_error"])
        logger.info("Indexing started for %s (pk=%s)", dbp.name, dbp.pk)

        from scripts.ingest import run_ingest

        t0 = time.time()
        stats = run_ingest(drop=True, settings=build_rag_settings(dbp, None))
        close_old_connections()
        dbp.refresh_from_db()
        dbp.index_status = DatabaseProfile.IndexStatus.READY
        dbp.indexed_vectors = stats["vectors_indexed"]
        dbp.indexed_at = timezone.now()
        dbp.index_error = ""
        dbp.save(update_fields=["index_status", "indexed_vectors", "indexed_at", "index_error"])
        logger.info(
            "Indexing finished for %s: %s tables, %s vectors in %.1fs",
            dbp.name, stats["tables"], stats["vectors_indexed"], time.time() - t0,
        )
    except Exception as exc:
        logger.error(
            "Indexing failed for pk=%s: %s\n%s", dbp_id, exc, traceback.format_exc()
        )
        close_old_connections()
        try:
            dbp = DatabaseProfile.objects.get(pk=dbp_id)
            dbp.index_status = DatabaseProfile.IndexStatus.ERROR
            dbp.index_error = f"{exc}\n\n{traceback.format_exc()}"[:4000]
            dbp.save(update_fields=["index_status", "index_error"])
        except Exception:
            logger.critical("Could not persist index error state for pk=%s", dbp_id, exc_info=True)
