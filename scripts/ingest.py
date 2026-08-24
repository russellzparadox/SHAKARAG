from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import build_chunks, dump_catalog
from rag.config import load_settings
from rag.db import connect
from rag.dialects import get_dialect
from rag.embeddings import get_embedder
from rag.enrich import fetch_odoo_metadata
from rag.store import VectorStore


def run_ingest(
    drop: bool = False, catalog_path: Path | None = None, settings: Any = None
) -> dict[str, Any]:
    t0 = time.time()
    if settings is None:
        settings = load_settings()
    dialect = get_dialect(settings)

    tables_map = dialect.introspect()
    tables = list(tables_map.values())
    n_cols = sum(len(t.columns) for t in tables)

    models_by_table: dict = {}
    fields_by_key: dict = {}
    if dialect.supports_odoo_metadata:
        conn = connect(settings)
        try:
            models_by_table, fields_by_key = fetch_odoo_metadata(conn)
        finally:
            conn.close()
    matched = sum(1 for t in tables if t.name in models_by_table)

    from rag.warehouse import classify_tables, candidate_value_columns

    role_counts = classify_tables(tables)
    sampled = 0
    if settings.sample_values and hasattr(dialect, "sample_values"):
        col_by_name: dict[tuple[str, str, str], Any] = {}
        for rec in tables:
            for c in rec.columns:
                col_by_name[(rec.schema, rec.name, c.name)] = c
        for rec in tables:
            if rec.kind != "r" or not (0 < rec.row_estimate < settings.value_sample_max_rows):
                continue
            for col_name in candidate_value_columns(rec):
                values = dialect.sample_values(rec.schema, rec.name, col_name, k=10)
                if values:
                    col_by_name[(rec.schema, rec.name, col_name)].sample_values = values
                    sampled += 1

    chunks = build_chunks(tables, models_by_table, fields_by_key, engine_label=dialect.label)

    if catalog_path:
        dump_catalog(tables, catalog_path)

    embedder = get_embedder(settings)
    store = VectorStore(str(settings.chroma_dir), settings.collection, embedder)

    existing = store.count
    dropped = False
    if drop and existing:
        store.reset()
        dropped = True

    def progress(done: int, total: int) -> None:
        if done % (total // 10 + 1) == 0 or done == total:
            pct = int(done * 100 / max(total, 1))
            print(f"  embedding/upserting: {done}/{total} ({pct}%)")

    store.upsert(chunks, progress=progress)

    elapsed = time.time() - t0
    return {
        "dialect": dialect.label,
        "tables": len(tables),
        "columns": n_cols,
        "warehouse_roles": role_counts,
        "value_samples": sampled,
        "odoo_models": len(models_by_table),
        "models_matched": matched,
        "field_descriptions": len(fields_by_key),
        "chunks": len(chunks),
        "vectors_before": existing,
        "dropped_existing": dropped,
        "vectors_indexed": store.count,
        "embedder": getattr(embedder, "tag", "custom"),
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the live database schema into the RAG index.")
    parser.add_argument("--drop", action="store_true", help="Rebuild the collection from scratch.")
    parser.add_argument("--dump-catalog", type=Path, default=None, help="Also write a JSON catalog of the schema.")
    args = parser.parse_args()

    settings = load_settings()
    print(f"Connecting ({settings.db_dialect}) to {settings.db_user}@{settings.db_host}:{settings.db_port}/{settings.db_name} ...")
    print("Introspecting schema ...")
    stats = run_ingest(drop=args.drop, catalog_path=args.dump_catalog)
    print(
        f"  [{stats['dialect']}] {stats['tables']} tables/views · {stats['columns']} columns · "
        f"{stats['chunks']} chunks"
    )
    if stats["odoo_models"]:
        print(
            f"  Odoo metadata: {stats['models_matched']}/{stats['odoo_models']} models matched, "
            f"{stats['field_descriptions']} field descriptions"
        )
    roles = stats.get("warehouse_roles") or {}
    if any(v for k, v in roles.items() if k != "unknown"):
        print(f"  Warehouse roles: {roles}")
    if stats.get("value_samples"):
        print(f"  Sampled values for {stats['value_samples']} columns")
    print(f"  Embedder: {stats['embedder']}")
    print(f"Indexed {stats['vectors_indexed']} vectors in {stats['elapsed_seconds']}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
