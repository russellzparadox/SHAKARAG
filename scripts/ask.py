from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import format_markdown_table, make_pipeline
from rag.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the schema RAG a natural-language question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="Show retrieval context only; no LLM or SQL execution.")
    parser.add_argument("--no-exec", action="store_true", help="Generate SQL but do not run it.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved schema context.")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    question = " ".join(args.question)
    settings = load_settings()
    pipeline = make_pipeline(settings)

    result = pipeline.ask(question, execute=not args.no_exec, dry_run=args.dry_run, top_k=args.top_k)

    if args.as_json:
        payload = {
            "question": result.question,
            "sql": result.sql,
            "explanation": result.explanation,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "tables_used": result.tables_used,
            "answer": result.answer,
            "error": result.error,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQ: {question}\n")

    if args.show_context:
        print("=" * 70)
        print("RETRIEVED CONTEXT")
        print("=" * 70)
        print(result.context[:4000])
        print()

    if result.tables_used and not args.show_context:
        print("Tables retrieved:", ", ".join(result.tables_used))

    if args.dry_run:
        print("\n(dry run: context above would be sent to the LLM)")
        return 0

    if result.sql:
        print("-" * 70)
        print("SQL:")
        print(result.sql)
        if result.explanation:
            print(f"\nWhy: {result.explanation}")

    if result.error and not result.rows:
        print(f"\nERROR: {result.error}")

    if result.rows is not None and len(result.columns):
        print("\nResults:")
        print(format_markdown_table(result.columns, result.rows))
        suffix = f" (capped at {settings.max_rows})" if result.truncated else ""
        print(f"{result.row_count} rows{suffix}")

    if result.answer:
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result.answer)
    elif not result.error:
        print("\n(no answer synthesized)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
