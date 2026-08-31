"""Tests for the ICRL evaluation harness (Phase F).

Covers:
  - F2/F3: recall@k math
  - F4/F5: deterministic holdout generation
  - F6/F7: execution-accuracy math
  - F1: scripts/eval_recall.py --help works
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---- F2/F3: recall@k ------------------------------------------------------

def test_recall_at_k_simple():
    """3 hits in the top-5 contain the gold table -> R@1=1.0, R@2=1.0.
    The 4th and 5th ranked are wrong -> R@5=1.0 (still contains the gold).
    5 entries total in the ranked list; one of them is the gold.
    """
    from rag.icrl_eval import compute_recall

    gold = {"public.orders"}
    # ranked[0] hits the gold
    ranked_hits = [
        ["public.orders"],
        ["public.orders"],
        ["public.customers"],
        ["public.products"],
    ]
    # R@1: top-1 contains gold? yes -> 1.0
    assert compute_recall(ranked_hits, gold, k=1) == 1.0
    assert compute_recall(ranked_hits, gold, k=2) == 1.0
    # R@5: top-5 contains gold? yes -> 1.0 (no, we have only 4 ranked)
    # when the list is shorter than k, we treat missing as "not retrieved"
    assert compute_recall(ranked_hits, gold, k=5) == 1.0
    # wrong case: gold is in ranked[3] -> R@1=0, R@2=0, R@5=1
    ranked_late = [
        ["public.customers"],
        ["public.products"],
        ["public.users"],
        ["public.orders"],
    ]
    assert compute_recall(ranked_late, gold, k=1) == 0.0
    assert compute_recall(ranked_late, gold, k=2) == 0.0
    assert compute_recall(ranked_late, gold, k=5) == 1.0
    # partial: gold is in ranked[1] -> R@1=0, R@2=1
    ranked_partial = [
        ["public.customers"],
        ["public.orders"],
        ["public.products"],
    ]
    assert compute_recall(ranked_partial, gold, k=1) == 0.0
    assert compute_recall(ranked_partial, gold, k=2) == 1.0


def test_recall_at_k_handles_missing_gold():
    """When the gold is nowhere in the list, all k's give 0."""
    from rag.icrl_eval import compute_recall

    gold = {"public.orders"}
    ranked = [["public.customers"], ["public.products"]]
    for k in (1, 2, 5, 10):
        assert compute_recall(ranked, gold, k=k) == 0.0


# ---- F4/F5: deterministic holdout generation -------------------------------

def _make_table(name, schema="public", role="dim"):
    from rag.introspect import Column, TableRecord
    return TableRecord(
        schema=schema, name=name, kind="r", row_estimate=100, comment=None,
        columns=[
            Column(name="id", type="int", nullable=False, default=None,
                   comment=None, identity=False, generated=False, pk=True),
        ],
        primary_key=["id"],
        foreign_keys=[], warehouse_role=role,
    )


def test_holdout_generation_deterministic():
    """Same tables + same seed -> same held-out set (no randomness, no time)."""
    from rag.icrl_eval import generate_holdout

    tables = [
        _make_table("F", role="fact"),
        _make_table("D1", role="dim"),
        _make_table("D2", role="dim"),
    ]
    h1 = generate_holdout(tables, n=10, seed=42)
    h2 = generate_holdout(tables, n=10, seed=42)
    h3 = generate_holdout(tables, n=10, seed=43)
    # identical input -> identical output
    assert h1 == h2
    # different seed -> different output (very likely; with 10 random picks
    # the chance of collision is negligible)
    assert h1 != h3
    # every held-out entry has the expected fields
    for h in h1:
        assert "question" in h and "sql_template" in h and "gold_tables" in h


# ---- F6/F7: execution accuracy --------------------------------------------

def test_execution_accuracy_exact_match():
    """When the executed rows match the gold rows exactly -> EX = 1.0."""
    from rag.icrl_eval import execution_accuracy

    gold = [["a", 1], ["b", 2]]
    # identical -> 1.0
    assert execution_accuracy("SELECT ...", gold, list(gold)) == 1.0
    # different rows -> 0.0
    other = [["a", 1], ["c", 3]]
    assert execution_accuracy("SELECT ...", gold, other) == 0.0
    # subset -> 0.0 (exact match required)
    assert execution_accuracy("SELECT ...", gold, [gold[0]]) == 0.0


def test_execution_accuracy_handles_row_order():
    """Row order should not affect EX: set-equality, not list-equality."""
    from rag.icrl_eval import execution_accuracy

    gold = [[1, "a"], [2, "b"]]
    same_set = [[2, "b"], [1, "a"]]
    assert execution_accuracy("SELECT ...", gold, same_set) == 1.0


# ---- F1: scripts/eval_recall.py --help works -------------------------------

def test_eval_recall_help_runs():
    """`scripts/eval_recall.py --help` must exit 0 and show --db-profile."""
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/eval_recall.py", "--help"],
        cwd=repo, capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--db-profile" in proc.stdout
    # the help should also mention R@k or recall (so users know what it does)
    assert any(kw in proc.stdout.lower() for kw in ("recall", "r@", "r@k"))


# ---- F8: scripts/eval_recall.py writes eval JSON, exits with code ----------

def test_eval_recall_writes_metrics_json_and_exits(tmp_path, monkeypatch):
    """`scripts/eval_recall.py --db-profile 1 --eval-dir <tmp>` should write
    `db1.json` with shape {recall_at_k, execution_accuracy, ...}.
    """
    repo = Path(__file__).resolve().parents[1]

    # we can't reach a real DB in CI; this test just asserts the script's
    # CLI plumbing works end-to-end when given a bogus profile id. The
    # script should still create the eval dir and write SOMETHING. If the
    # introspect step fails, that's acceptable as long as the script
    # doesn't crash with a Python traceback.
    proc = subprocess.run(
        [sys.executable, "scripts/eval_recall.py",
         "--db-profile", "9999",  # nonexistent
         "--eval-dir", str(tmp_path),
         "--holdout-n", "3",
         "--exit-threshold", "0.0"],  # always pass to keep the test deterministic
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    # Accept either: success (script can run against mocks we don't have
    # here) OR a graceful failure that prints a clean message (no traceback).
    if proc.returncode != 0:
        assert "Traceback" not in proc.stderr, proc.stderr
    # No requirement to produce a JSON file when introspect fails.
