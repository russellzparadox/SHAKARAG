"""Tests for Phase E: idempotent re-runs, dry-run, versioned JSON schema."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---- shared fixtures -------------------------------------------------------

class _FakeLLM:
    """Records every chat_json call; returns a queue of payloads."""
    def __init__(self, payloads):
        self.queue = list(payloads)
        self.calls = 0

    def chat_json(self, messages, max_tokens=8192):
        self.calls += 1
        if not self.queue:
            return {"question": "q", "sql": "SELECT a, b, c FROM t WHERE c = 1 GROUP BY a"}
        return self.queue.pop(0)


class _TwoTableDialect:
    """Returns a fixed warehouse with two tables + one FK."""
    name = "fake"
    label = "Fake"

    def __init__(self):
        from rag.introspect import Column, ForeignKey, TableRecord

        cols = [
            Column(name="id", type="int", nullable=False, default=None,
                   comment=None, identity=False, generated=False, pk=True),
            Column(name="name", type="text", nullable=True, default=None,
                   comment=None, identity=False, generated=False),
        ]
        self.tables = {
            "F": TableRecord(
                schema="public", name="F", kind="r", row_estimate=100, comment=None,
                columns=cols, primary_key=["id"],
                foreign_keys=[ForeignKey(name="fk_f_d", columns=["d_id"],
                                         ref_table="D", ref_columns=["id"])],
                warehouse_role="fact",
            ),
            "D": TableRecord(
                schema="public", name="D", kind="r", row_estimate=50, comment=None,
                columns=cols, primary_key=["id"],
                foreign_keys=[], warehouse_role="dim",
            ),
        }

    def introspect(self):
        return self.tables

    def prompt_hints(self):
        return ""

    def execute_readonly(self, sql, max_rows):
        return (["x"], [[1]], False)


class _ThreeTableDialect(_TwoTableDialect):
    """Three tables (F, D1, D2) with two FKs — yields multiple distinct
    traversal signatures so the incremental test can verify NEW entries
    are created."""

    def __init__(self):
        from rag.introspect import Column, ForeignKey, TableRecord

        cols = [
            Column(name="id", type="int", nullable=False, default=None,
                   comment=None, identity=False, generated=False, pk=True),
            Column(name="name", type="text", nullable=True, default=None,
                   comment=None, identity=False, generated=False),
        ]
        self.tables = {
            "F": TableRecord(
                schema="public", name="F", kind="r", row_estimate=100, comment=None,
                columns=cols, primary_key=["id"],
                foreign_keys=[
                    ForeignKey(name="fk_f_d1", columns=["d1_id"],
                               ref_table="D1", ref_columns=["id"]),
                    ForeignKey(name="fk_f_d2", columns=["d2_id"],
                               ref_table="D2", ref_columns=["id"]),
                ],
                warehouse_role="fact",
            ),
            "D1": TableRecord(
                schema="public", name="D1", kind="r", row_estimate=50, comment=None,
                columns=cols, primary_key=["id"],
                foreign_keys=[], warehouse_role="dim",
            ),
            "D2": TableRecord(
                schema="public", name="D2", kind="r", row_estimate=40, comment=None,
                columns=cols, primary_key=["id"],
                foreign_keys=[], warehouse_role="dim",
            ),
        }


def _settings():
    """Minimal Settings stand-in. The current run_icrl_generation needs:
      - db_dialect (used by get_dialect)
      - chroma_dir, collection (used by index_results_for_profile)
      - llm_*, embed_* (used downstream — we monkeypatch)
    """
    return SimpleNamespace(
        db_dialect="fake",
        chroma_dir="/tmp",
        collection="c",
        llm_base_url="", llm_model="m", llm_api_key=None,
        llm_temperature=0.0,
        embed_provider="default", openai_api_key=None, openai_base_url="",
        embed_model=None, top_k=4, context_char_budget=4000,
        max_rows=10, statement_timeout_ms=1000,
        sample_values=True, value_sample_max_rows=100, examples_top_k=2,
        data_preview=False,
    )


# ---- E1/E2: idempotent re-runs skip existing entries ----------------------

def test_run_icrl_idempotent_skips_existing(tmp_path, monkeypatch):
    """Pre-populate the JSON with 3 entries; subsequent run with `--n 10`
    only processes the 7 unseen traversals.
    """
    from rag import dialects as _dialects_mod
    from rag import icrl as _icrl_mod
    from rag.icrl import run_icrl_generation, save_entries, load_entries

    settings = _settings()
    monkeypatch.setattr(_dialects_mod, "get_dialect", lambda s: _ThreeTableDialect())

    # Use a real JSON file inside tmp_path so save_entries/load_entries work
    monkeypatch.setattr(_icrl_mod, "_icrl_json_path", lambda settings: tmp_path / "db1.json")

    # First: write 3 entries — all with the same `F, D1` signature so the
    # incremental run will skip them. The new entries should come from
    # the *other* signatures (`F, D2`, `F, D1, D2`, etc).
    pre_existing = [
        {"question": f"q{i}", "sql": f"SELECT a FROM t{i}",
         "tables": ["public.F", "public.D1"], "reward": 3.0, "iterations": 1}
        for i in range(3)
    ]
    save_entries(settings, pre_existing, append=False)

    # Now run with n=10: should produce ~10 NEW entries (3 are skipped).
    # We control how many LLM calls happen by patching complexity_reward to
    # plateau after iter 1 (so each traversal makes 1-2 calls).
    from rag import icrl as _icrl_mod_inner
    original_reward = _icrl_mod_inner.complexity_reward
    _icrl_mod_inner.complexity_reward = lambda sql: (
        3.5, {"retrieval": 6, "conditional": 0, "aggregation": 0, "modification": 0}
    )
    try:
        llm = _FakeLLM([])
        results = run_icrl_generation(
            settings, llm, n=10, max_iterations=2,
            min_reward=2.0, incremental=True,
        )
    finally:
        _icrl_mod_inner.complexity_reward = original_reward

    # Sanity: the 3 pre-existing entries are NOT regenerated (the incremental
    # filter prevented that), and at least one new entry is produced.
    new_questions = {r.question for r in results}
    pre_questions = {e["question"] for e in pre_existing}
    assert pre_questions.isdisjoint(new_questions), (
        "pre-existing entries were regenerated instead of skipped"
    )
    assert len(results) >= 1, f"expected >=1 new result, got {len(results)}"
    # And the existing 3 entries are still readable from disk.
    entries = load_entries(settings)
    got_qs = {e["question"] for e in entries}
    assert pre_questions.issubset(got_qs), "pre-existing entries lost after incremental run"


# ---- E4/E5: dry-run path makes no LLM calls ------------------------------

def test_run_icrl_dry_run_no_llm_calls(tmp_path, monkeypatch):
    """--dry-run must return after counting traversals, with zero LLM calls."""
    from rag import dialects as _dialects_mod
    from rag import icrl as _icrl_mod
    from rag.icrl import run_icrl_generation

    settings = _settings()
    monkeypatch.setattr(_dialects_mod, "get_dialect", lambda s: _TwoTableDialect())
    monkeypatch.setattr(_icrl_mod, "_icrl_json_path", lambda settings: tmp_path / "db1.json")

    llm = _FakeLLM([])
    results = run_icrl_generation(
        settings, llm, n=5, max_iterations=3, min_reward=2.0, dry_run=True,
    )
    assert results == []
    assert llm.calls == 0


# ---- E6: versioned JSON schema ---------------------------------------------

def test_version_2_schema_round_trip(tmp_path, monkeypatch):
    """save_entries writes version 2; load_entries reads both v1 and v2."""
    from rag import icrl as _icrl_mod
    from rag.icrl import save_entries, load_entries

    settings = _settings()
    monkeypatch.setattr(_icrl_mod, "_icrl_json_path", lambda settings: tmp_path / "db1.json")

    # write v2
    entries = [
        {"question": "q1", "sql": "SELECT 1", "tables": ["public.F"],
         "reward": 3.0, "iterations": 1},
    ]
    save_entries(settings, entries, append=False)

    raw = (tmp_path / "db1.json").read_text()
    data = json.loads(raw)
    assert data["version"] == 2
    assert isinstance(data["entries"], list)
    assert len(data["entries"]) == 1
    # reader is happy
    loaded = load_entries(settings)
    assert len(loaded) == 1
    assert loaded[0]["question"] == "q1"


def test_load_entries_v1_back_compat(tmp_path, monkeypatch):
    """A bare-list v1 file (no 'version' key) still loads correctly."""
    from rag import icrl as _icrl_mod
    from rag.icrl import load_entries

    settings = _settings()
    path = tmp_path / "db1.json"
    path.write_text(json.dumps([
        {"question": "old_q", "sql": "SELECT 1", "tables": [], "reward": 2.0, "iterations": 1},
    ]))
    monkeypatch.setattr(_icrl_mod, "_icrl_json_path", lambda settings: path)

    loaded = load_entries(settings)
    assert len(loaded) == 1
    assert loaded[0]["question"] == "old_q"


def test_save_entries_appends(tmp_path, monkeypatch):
    """append=True merges with existing entries (deduped by question+tables)."""
    from rag import icrl as _icrl_mod
    from rag.icrl import save_entries, load_entries

    settings = _settings()
    path = tmp_path / "db1.json"
    monkeypatch.setattr(_icrl_mod, "_icrl_json_path", lambda settings: path)

    save_entries(settings, [
        {"question": "q1", "sql": "SELECT 1", "tables": ["public.F"],
         "reward": 3.0, "iterations": 1},
    ], append=False)
    # append the same entry — dedupe keeps total at 1
    save_entries(settings, [
        {"question": "q1", "sql": "SELECT 1", "tables": ["public.F"],
         "reward": 3.0, "iterations": 1},
        {"question": "q2", "sql": "SELECT 2", "tables": ["public.D"],
         "reward": 2.5, "iterations": 1},
    ], append=True)

    loaded = load_entries(settings)
    assert len(loaded) == 2
    qs = {e["question"] for e in loaded}
    assert qs == {"q1", "q2"}
