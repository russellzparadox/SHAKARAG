"""Tests for the faithful reward + coach loop (paper §3.1, §A.3)."""
from __future__ import annotations

import json

import pytest

from rag.icrl import (
    ICRLGenerator,
    Traversal,
    bucket_gaps,
    complexity_reward,
    operator_suggestions,
)
from test_icrl import _table  # shared fixture helper (tests/ on sys.path under pytest)


# ---- B1: reward bucket counts match paper §A.3 ----

# Reference SQL from the plan, the same one the paper uses to demonstrate
# keyword weighting (§A.3 and Table 1's reference schema).
REFERENCE_SQL = (
    "SELECT d.title, SUM(f.amount) AS total FROM f "
    "JOIN d ON f.sid = d.sid WHERE f.year = 2024 "
    "GROUP BY d.title HAVING SUM(f.amount) > 100 ORDER BY total DESC"
)


def test_reward_bucket_normalisation_parity_with_paper_a3():
    """The counts returned by complexity_reward must match paper §A.3 exactly.

    Expected per-bucket values (paper §A.3):
      retrieval   = SELECT(1) + FROM(1) + JOIN(2) + ON(2) + WHERE(2)
                    + GROUP BY(3) + HAVING(3) + ORDER BY(2) = 16
      conditional = 0   (no AND/OR/NOT/IN/BETWEEN/LIKE/CASE present)
      aggregation = SUM(2×3) + DESC(1) = 7
      modification = 0
    """
    _, counts = complexity_reward(REFERENCE_SQL)
    assert counts["retrieval"] == 16
    assert counts["conditional"] == 0
    assert counts["aggregation"] == 7
    assert counts["modification"] == 0


# ---- B3: bucket_gaps orders buckets by weakness ----

def test_bucket_gaps_orders_by_weakest():
    """bucket_gaps returns buckets sorted by ascending count (weakest first)."""
    # `SELECT a FROM t` has only retrieval: SELECT(1) + FROM(1) = 2
    # all other buckets are 0
    counts = {"retrieval": 2, "conditional": 0, "aggregation": 0, "modification": 0}
    gaps = bucket_gaps(counts)
    # the three zero-count buckets come first (any order among equals is OK),
    # then retrieval
    assert gaps[-1] == "retrieval"
    assert set(gaps) == {"retrieval", "conditional", "aggregation", "modification"}
    # and within the zero-bucket group, modification is penalised so it should
    # be ordered to come BEFORE the other zeros (negative weight on the
    # read-only system)
    assert gaps.index("modification") < gaps.index("aggregation")


# ---- B5: feedback prompt carries bucket gaps + operator suggestions ----

def test_feedback_prompt_contains_bucket_gaps_and_suggestions():
    """The feedback user-message must include the bucket-gaps JSON and
    operator-suggestion text so the coach LLM knows what to suggest."""
    captured: list[list[dict]] = []

    class _CaptureLLM:
        def __init__(self, payload):
            self.payload = payload
            self.last_messages = None

        def chat_json(self, messages, max_tokens=8192):
            self.last_messages = messages
            captured.append(messages)
            return self.payload

    from rag.icrl import _feedback, FEEDBACK_SYSTEM  # private API for test

    llm = _CaptureLLM({"feedback": "add a HAVING filter and a JOIN"})
    # two tables for the traversal fixture
    t1 = _table("F", role="fact", fks=["D"])
    t2 = _table("D")
    tr = Traversal(tables=[t1, t2], label="F -> D")
    counts = {"retrieval": 4, "conditional": 0, "aggregation": 1, "modification": 0}

    _feedback(llm, tr, "what is the sum by dim?", "SELECT f.a, SUM(f.b) FROM f GROUP BY f.a", counts)

    user_msg = llm.last_messages[1]["content"]
    # bucket gaps (as JSON) must be present
    assert "aggregation" in user_msg
    assert "conditional" in user_msg
    # operator-suggestion text for the weak buckets
    assert "JOIN" in user_msg or "GROUP BY" in user_msg or "HAVING" in user_msg
    # and the suggestion table from paper §A.3 should make at least one
    # operator-level hint appear (e.g. "CASE", "LEFT JOIN", "BETWEEN", "IN")
    suggestions_blob = " ".join(operator_suggestions.values()).upper()
    assert any(kw in user_msg.upper() for kw in ["JOIN", "GROUP BY", "HAVING", "CASE", "BETWEEN"])


# ---- B7: plateau termination ----

class _SeqLLM:
    """Returns a queued list of (gen, feedback) JSON payloads in order.

    Each call to chat_json pops the next entry. Used to drive ICRLGenerator
    deterministically through multiple iterations.
    """

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls: list[list[dict]] = []

    def chat_json(self, messages, max_tokens=8192):
        self.calls.append(messages)
        return self.queue.pop(0)


def _simple_traversal():
    t1 = _table("F", role="fact", fks=["D"])
    t2 = _table("D")
    return Traversal(tables=[t1, t2], label="F -> D")


def test_plateau_termination():
    """If reward improvement < plateau_epsilon, stop iterating early.

    Rewards: 1.0, 2.0, 2.1 with plateau_epsilon=0.5 → stop at iter 3
    (Δ=0.1 < 0.5, plateau detected after iter 3 produced no meaningful gain).
    """
    # iter1: gen returns SQL that scores 1.0; feedback returns "" (success)
    # iter2: gen SQL that scores 2.0; feedback returns ""
    # iter3: gen SQL that scores 2.1; feedback should NOT be requested
    payloads = [
        {"question": "q1", "sql": "SELECT a, b FROM t"},
        {"feedback": "add a join"},
        {"question": "q2", "sql": "SELECT a, b, c FROM t WHERE c = 1"},
        {"feedback": "add aggregation"},
        {"question": "q3", "sql": "SELECT a, b, c, d FROM t WHERE c = 1 AND d = 2"},
    ]
    # we need the *reward* to be controllable. Patch complexity_reward to
    # return the desired score based on the question suffix.
    from rag import icrl

    original = icrl.complexity_reward
    rewards = iter([1.0, 2.0, 2.1])

    def _patched(sql):
        r = next(rewards)
        # also return a stable counts dict so bucket_gaps doesn't crash
        return r, {"retrieval": int(r), "conditional": 0, "aggregation": 0, "modification": 0}

    icrl.complexity_reward = _patched
    try:
        llm = _SeqLLM(payloads)
        gen = ICRLGenerator(llm, max_iterations=10, plateau_epsilon=0.5)
        result = gen.run(_simple_traversal(), min_reward=10.0)  # never reached by reward
        # iterations must stop at 3 (plateau detected)
        assert result.iterations == 3, f"expected stop at 3, got {result.iterations}"
        # only 5 chat_json calls (3 gen + 2 feedback — no feedback for iter 3)
        assert len(llm.calls) == 5, f"expected 5 calls, got {len(llm.calls)}"
    finally:
        icrl.complexity_reward = original


# ---- B9: max_iterations caps the loop ----

def test_max_iterations_caps_plateau_loop():
    """Constant reward → plateau detected immediately, but max_iterations still caps total iters."""
    payloads = [
        {"question": f"q{i}", "sql": "SELECT a FROM t"}
        for i in range(1, 6)  # 5 iters × (1 gen + 1 feedback) = 10 calls
    ]
    # we only need 3 max iters; plateau also kicks in but max wins
    payloads = payloads[:6]  # 3 gen + 3 feedback
    from rag import icrl

    original = icrl.complexity_reward
    icrl.complexity_reward = lambda sql: (
        1.0, {"retrieval": 1, "conditional": 0, "aggregation": 0, "modification": 0}
    )
    try:
        llm = _SeqLLM(payloads)
        gen = ICRLGenerator(llm, max_iterations=3, plateau_epsilon=0.5)
        result = gen.run(_simple_traversal(), min_reward=10.0)
        # either plateau fires after iter 1 (no improvement) or max_iter caps
        assert result.iterations <= 3
    finally:
        icrl.complexity_reward = original


# ---- B10: malformed SQL is rejected before reward ----

def test_executable_sql_required_for_reward_score():
    """Malformed SQL must not be scored; iteration is retried with feedback.

    A parse error from sqlglot should set a parse-error feedback context
    and continue the loop instead of crashing or computing a misleading
    high reward via keyword density.
    """
    payloads = [
        # iter1: malformed → parse fails, retry
        {"question": "bad", "sql": "SELECT a FROM JOIN b ON x ="},
        # iter2: well-formed
        {"question": "good", "sql": "SELECT a, b FROM t WHERE a = 1"},
    ]
    from rag import icrl

    original_complexity = icrl.complexity_reward
    complexity_calls: list[str] = []
    def _track(sql):
        complexity_calls.append(sql)
        return 1.5, {"retrieval": 3, "conditional": 0, "aggregation": 0, "modification": 0}
    icrl.complexity_reward = _track
    try:
        llm = _SeqLLM(payloads)
        gen = ICRLGenerator(llm, max_iterations=3, plateau_epsilon=0.5)
        result = gen.run(_simple_traversal(), min_reward=10.0)
        # malformed SQL was NOT scored
        assert all("JOIN b" not in c for c in complexity_calls), \
            f"malformed SQL reached complexity_reward: {complexity_calls}"
        # well-formed SQL was scored
        assert any("WHERE a = 1" in c for c in complexity_calls)
        # result is the well-formed one
        assert result is not None
        assert "WHERE a = 1" in result.sql
    finally:
        icrl.complexity_reward = original_complexity
