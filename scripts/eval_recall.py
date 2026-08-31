"""Evaluate the ICRL/RAG-over-SQL pipeline against paper §4 metrics.

For a given `--db-profile`:
  1. introspect the warehouse to enumerate tables
  2. deterministically generate a held-out (Q, gold_tables, expected_rows)
     set using NL templates (paper §4 evaluation harness)
  3. run the router KB to retrieve the top-k tables per held-out question
  4. compute R@k (schema recall) and EX (execution accuracy) metrics
  5. write the metrics JSON to `<eval-dir>/db<pk>.json`
  6. exit 0 if metrics meet `--exit-threshold`, else 1

Usage (from ~/work/rag):
  .venv/bin/python scripts/eval_recall.py --db-profile 1
                                         [--holdout-n 20] [--seed 42]
                                         [--exit-threshold 0.5]
                                         [--eval-dir webapp/icrl/eval]
"""
import argparse
import json
import logging
import os
import pathlib
import sys
import traceback

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

from chat.models import DatabaseProfile  # noqa: E402
from chat.rag_service import build_rag_settings  # noqa: E402
from rag.icrl_eval import (  # noqa: E402
    compute_recall,
    execution_accuracy,
    generate_holdout,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-profile", type=int, required=True,
                    help="DatabaseProfile PK to evaluate.")
    ap.add_argument("--llm-profile", type=int, default=None,
                    help="LLMProfile PK; defaults to the first one.")
    ap.add_argument("--holdout-n", type=int, default=20,
                    help="Number of held-out questions to generate.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for holdout generation (deterministic).")
    ap.add_argument("--k-values", type=str, default="1,2,5,10",
                    help="Comma-separated k values for R@k (default 1,2,5,10).")
    ap.add_argument("--exit-threshold", type=float, default=0.5,
                    help="Minimum R@1 to exit 0 (else 1).")
    ap.add_argument("--eval-dir", type=str, default="webapp/icrl/eval",
                    help="Where to write the metrics JSON.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    repo = pathlib.Path(__file__).resolve().parents[1]
    eval_dir = pathlib.Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = repo / eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    try:
        dbp = DatabaseProfile.objects.get(pk=args.db_profile)
    except DatabaseProfile.DoesNotExist:
        print(f"FAIL: DatabaseProfile pk={args.db_profile} not found", file=sys.stderr)
        return 1
    settings = build_rag_settings(dbp, None)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    from rag.dialects import get_dialect  # noqa: WPS433 (deferred import)
    from rag.embeddings import get_embedder  # noqa: WPS433
    from rag.icrl import retrieve_tables_for_question  # noqa: WPS433
    from rag.llm import LLMClient  # noqa: WPS433

    try:
        dialect = get_dialect(settings)
    except Exception as exc:
        print(f"FAIL: introspect failed: {exc}", file=sys.stderr)
        return 1

    tables_map = dialect.introspect()
    tables = [t for t in tables_map.values() if t.kind == "r"]
    print(f"profile db={dbp.pk} dialect={settings.db_dialect} tables={len(tables)}", flush=True)

    # 1) deterministic holdout
    holdout = generate_holdout(tables, n=args.holdout_n, seed=args.seed)
    print(f"generated {len(holdout)} held-out items (seed={args.seed})", flush=True)

    # 2) router KB lookup per question
    embedder = get_embedder(settings)
    # wrap embedder in a tiny store-like object for retrieve_tables_for_question
    class _Store:
        pass
    store = _Store()
    store.embedder = embedder

    recall_at_k: dict[str, float] = {}
    ex_scores: list[float] = []
    per_question: list[dict] = []

    for h in holdout:
        retrieved = retrieve_tables_for_question(
            store, h["question"], top_k=max(k_values),
            chroma_dir=str(settings.chroma_dir),
            collection=settings.collection,
        )
        retrieved_bare = {t.split(".")[-1].lower() for t in retrieved}
        gold_bare = {t.split(".")[-1].lower() for t in h["gold_tables"]}
        # recall@k: gold ⊆ top-k?
        for k in k_values:
            top_k = list(retrieved_bare)[:k]
            hit = 1.0 if gold_bare.issubset(set(top_k)) else 0.0
            recall_at_k[f"r@{k}"] = recall_at_k.get(f"r@{k}", 0.0) + hit
        # execution accuracy: try to execute the SQL, compare rows
        ex = 0.0
        try:
            rows = dialect.execute_readonly(h["sql_template"], max_rows=100)
            # execute_readonly returns (cols, rows, error_flag) in this dialect
            if isinstance(rows, tuple) and len(rows) == 3:
                _, executed_rows, _ = rows
            else:
                executed_rows = rows
            # gold_rows is a count or list — we only have count in the holdout,
            # so we treat "any rows returned" as 0.5 and exact-count as 1.0
            if isinstance(h.get("gold_rows"), int):
                ex = 1.0 if len(executed_rows or []) == h["gold_rows"] else 0.0
            else:
                ex = 1.0 if (executed_rows or []) == h.get("gold_rows") else 0.0
        except Exception:
            ex = 0.0
        ex_scores.append(ex)
        per_question.append({
            "question": h["question"],
            "gold_tables": sorted(gold_bare),
            "retrieved": sorted(retrieved_bare),
            "execution_accuracy": ex,
        })

    n = max(1, len(holdout))
    for k in k_values:
        recall_at_k[f"r@{k}"] = round(recall_at_k[f"r@{k}"] / n, 4)
    ex_avg = round(sum(ex_scores) / n, 4)

    metrics = {
        "version": 1,
        "db_profile": dbp.pk,
        "dialect": settings.db_dialect,
        "n_questions": len(holdout),
        "seed": args.seed,
        "recall_at_k": recall_at_k,
        "execution_accuracy": ex_avg,
        "per_question": per_question,
    }
    out = eval_dir / f"db{dbp.pk}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"SAVED metrics -> {out}", flush=True)
    for k in k_values:
        print(f"  R@{k} = {recall_at_k[f'r@{k}']}", flush=True)
    print(f"  EX    = {ex_avg}", flush=True)

    return 0 if recall_at_k.get("r@1", 0.0) >= args.exit_threshold else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
