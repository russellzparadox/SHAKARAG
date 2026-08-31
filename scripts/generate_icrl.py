#!/usr/bin/env python
"""Generate + index ICRL synthetic QA pairs for a database (paper method).

Usage:
    python scripts/generate_icrl.py [--n 30] [--iters 3] [--drop]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_settings
from rag.dialects import get_dialect
from rag.icrl import (
    ICRLGenerator,
    ROUTER_SUFFIX,
    build_schema_graph,
    index_qa_triplets,
    random_walk_traversals,
)
from rag.llm import LLMClient
from rag.sqlguard import validate_sql


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of synthetic QA triplets")
    ap.add_argument("--iters", type=int, default=3, help="max ICRL iterations per traversal")
    args = ap.parse_args()

    settings = load_settings()
    dialect = get_dialect(settings)
    print(f"introspecting {settings.db_dialect} ...")
    tables_map = dialect.introspect()
    tables = [t for t in tables_map.values() if t.kind == "r"]
    print(f"{len(tables)} tables")

    graph = build_schema_graph(tables)
    n_edges = sum(len(v) for v in graph.values())
    print(f"schema graph: {len(graph)} nodes / {n_edges} FK edges")

    traversals = random_walk_traversals(tables, n_walks=args.n)
    print(f"{len(traversals)} unique traversals")

    llm = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
    )
    gen = ICRLGenerator(llm, max_iterations=args.iters)

    results = []
    from rag.embeddings import get_embedder
    from rag.store import VectorStore

    embedder = get_embedder(settings)
    store = VectorStore(str(settings.chroma_dir), settings.collection, embedder)

    for i, tr in enumerate(traversals, start=1):
        print(f"[{i}/{len(traversals)}] {tr.label}")
        try:
            res = gen.run(tr, validate_sql=validate_sql)
        except Exception as exc:
            print(f"  skipped ({exc})")
            continue
        if res and res.reward >= 2.0:
            results.append(res)
            print(f"  reward={res.reward} iters={res.iterations} Q: {res.question[:90]}")

    # index into <collection>-router so reindexing the main collection keeps them separate
    class _RouterStore:
        pass

    router_store = _RouterStore()
    router_store.embedder = embedder
    router_store.collection = store._client.get_or_create_collection(
        name=settings.collection + ROUTER_SUFFIX,
        metadata={"embedder": embedder.tag, "kind": "router"},
        configuration={"hnsw": {"space": "cosine"}},
    )
    indexed = index_qa_triplets(router_store, results)
    print(f"indexed {indexed} routing triplets into '{settings.collection}{ROUTER_SUFFIX}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
