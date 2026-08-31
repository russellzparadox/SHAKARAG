"""Run ICRL synthetic-QA generation for a profile and index the routing KB.

Usage (from ~/work/rag):
  .venv/bin/python scripts/run_icrl.py [--db-profile 1] [--llm-profile 1]
                                       [--n 30] [--max-iterations 3] [--min-reward 2.0]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "webapp"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
os.environ.setdefault(
    "DJANGO_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "webapp", "db.sqlite3"),
)
os.chdir(os.path.join(os.path.dirname(__file__), "..", "webapp"))

import django  # noqa: E402

django.setup()

from chat.models import DatabaseProfile, LLMProfile  # noqa: E402
from chat.rag_service import build_rag_settings  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.icrl import (  # noqa: E402
    get_router_collection,
    index_results_for_profile,
    run_icrl_generation,
)
from rag.llm import LLMClient  # noqa: E402
from rag.sqlguard import validate_sql  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-profile", type=int, required=True)
    ap.add_argument("--llm-profile", type=int, default=None)
    ap.add_argument("--n", type=int, default=30, help="number of traversals")
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--min-reward", type=float, default=2.0)
    ap.add_argument(
        "--timeout", type=float, default=180.0,
        help="Seconds to wait per LLM request (default 180). Raise to allow slow "
             "router/models more time (e.g. 600).",
    )
    ap.add_argument(
        "--drop", action="store_true",
        help="Drop the <collection>-router KB first, then rebuild it from scratch.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    dbp = DatabaseProfile.objects.get(pk=args.db_profile)
    llmp = (
        LLMProfile.objects.get(pk=args.llm_profile)
        if args.llm_profile
        else LLMProfile.objects.first()
    )
    settings = build_rag_settings(dbp, llmp)
    print(f"db={dbp.name} ({settings.db_dialect}) collection={settings.collection} llm={llmp.name if llmp else '-'}", flush=True)

    llm = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout=args.timeout,
    )
    embedder = get_embedder(settings)

    if args.drop:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        rname = settings.collection + "-router"
        try:
            client.delete_collection(rname)
            print(f"DROPPED router KB '{rname}'", flush=True)
        except Exception as exc:
            print(f"router KB '{rname}' not dropped: {exc}", flush=True)

    results = run_icrl_generation(
        settings,
        llm,
        n=args.n,
        max_iterations=args.max_iterations,
        validate_sql=validate_sql,
        min_reward=args.min_reward,
    )
    print(f"GENERATED {len(results)}", flush=True)
    for r in results:
        print(f"  reward={r.reward} iters={r.iterations} Q: {r.question[:100]}", flush=True)

    if results:
        import json
        import pathlib

        out = pathlib.Path(os.environ["DJANGO_DB_PATH"]).parent / "icrl" / f"db{dbp.pk}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([
            {
                "question": r.question,
                "sql": r.sql,
                "tables": r.tables,
                "reward": r.reward,
                "iterations": r.iterations,
            }
            for r in results
        ], indent=2))
        print(f"SAVED {len(results)} -> {out}", flush=True)
        n = index_results_for_profile(settings, embedder, results)
        col = get_router_collection(settings, embedder)
        print(f"INDEXED {n} -> router count {col.count()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
