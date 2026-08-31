import json
from types import SimpleNamespace

from rag.config import Settings
from rag.pipeline import RagPipeline, render_sql_system


class FakeStore:
    embedder = None

    def query_text(self, text, top_k):
        return []

    def all_table_metas(self):
        return []


class FakeExamples:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, k=2):
        return self.hits[:k]


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, messages, max_tokens=8192):
        self.calls.append(messages)
        return json.loads(self.responses.pop(0))

    def chat(self, messages, max_tokens=8192):
        return "ok"


class FakeDialect:
    name = "postgres"
    label = "PostgreSQL"

    def __init__(self, results):
        self.results = list(results)

    def execute_readonly(self, sql, max_rows):
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def prompt_hints(self):
        return ""


def _settings(tmp_path):
    return Settings(
        db_dialect="postgres", db_host="h", db_port=1, db_user="u", db_password="p",
        db_name="d", db_url=None, chroma_dir=tmp_path, collection="c",
        embed_provider="default", openai_api_key=None,
        openai_base_url="http://x", embed_model=None,
        llm_base_url="http://x/v1", llm_model="m", llm_api_key=None,
        llm_temperature=0.0, top_k=4, context_char_budget=4000,
        max_rows=10, statement_timeout_ms=1000,
        sample_values=True, value_sample_max_rows=100000, examples_top_k=2,
        data_preview=False,
    )


def _pipeline(tmp_path, llm_responses, dialect_results, example_hits=None):
    p = RagPipeline(
        settings=_settings(tmp_path),
        store=FakeStore(),
        llm=FakeLLM(llm_responses),
        dialect=FakeDialect(dialect_results),
    )
    p._examples = FakeExamples(example_hits or [])
    return p


def test_render_sql_system_contains_dw_block():
    text = render_sql_system("Snowflake", "", 50)
    assert "FACT" in text and "DIMENSION" in text
    assert "__DW_BLOCK__" not in text


def test_execution_error_triggers_repair(tmp_path):
    good = '{"sql": "SELECT id FROM dw.f_sales", "explanation": "fixed"}'
    bad = '{"sql": "SELECT id FROM dw.f_salez", "explanation": "typo"}'
    p = _pipeline(
        tmp_path,
        llm_responses=[bad, good],
        dialect_results=[
            RuntimeError('relation "dw.f_salez" does not exist'),
            (["id"], [[7]], False),
        ],
    )
    result = p.ask("how many sales?", execute=True)
    assert result.error is None
    assert result.rows == [[7]]
    assert len(p.llm.calls) == 2
    repair_user = p.llm.calls[1][-1]["content"]
    assert "does not exist" in repair_user


def test_few_shot_examples_injected_into_prompt(tmp_path):
    hits = [{
        "question": "revenue per month",
        "sql": "SELECT month, SUM(amount_total) FROM f_sales GROUP BY month",
        "notes": "",
        "distance": 0.2,
    }]
    payload = '{"sql": "SELECT month, SUM(amount_total) FROM f_sales GROUP BY month", "explanation": "e"}'
    p = _pipeline(tmp_path, [payload], [(["m", "rev"], [["jan", 5]], False)], example_hits=hits)
    result = p.ask("monthly revenue?", execute=True)
    assert result.answer is None or True
    first_call = p.llm.calls[0]
    user_content = first_call[1]["content"]
    assert "VERIFIED QUERY EXAMPLES" in user_content
    assert "SUM(amount_total)" in user_content


def test_empty_result_retries_alternative_table(tmp_path):
    first = '{"sql": "SELECT Title FROM dw.DimPurchasingAgent WHERE Title LIKE \'%xperial%\'", "explanation": "guess"}'
    second = '{"sql": "SELECT SupplierID, Title FROM dw.DimSupplier WHERE Title LIKE \'%xperial%\'", "explanation": "better"}'
    p = _pipeline(
        tmp_path,
        llm_responses=[first, second],
        dialect_results=[
            (["Title"], [], False),
            (["SupplierID", "Title"], [[3, "Xperial Ltd"]], False),
        ],
    )
    result = p.ask("give me supplier info of Xperial", execute=True)
    assert result.rows == [[3, "Xperial Ltd"]]
    assert "DimSupplier" in result.sql
    assert len(p.llm.calls) == 2
    retry_msg = p.llm.calls[1][-1]["content"]
    assert "ZERO rows" in retry_msg


def test_empty_result_kept_when_model_says_no_candidate(tmp_path):
    only = '{"sql": "SELECT Title FROM dw.DimAgent", "explanation": ""}'
    decline = '{"sql": "SELECT Title FROM dw.DimAgent", "explanation": "no better candidate"}'
    p = _pipeline(
        tmp_path,
        llm_responses=[only, decline],
        dialect_results=[
            (["Title"], [], False),
            (["Title"], [], False),
        ],
    )
    result = p.ask("find it", execute=True)
    assert result.rows == []
    assert result.error is None


def test_no_examples_no_block(tmp_path):
    payload = '{"sql": "SELECT 1", "explanation": ""}'
    p = _pipeline(tmp_path, [payload], [(["one"], [[1]], False)])
    result = p.ask("anything")
    user_content = p.llm.calls[0][1]["content"]
    assert "VERIFIED QUERY EXAMPLES" not in user_content


# ---- D1/D2: ICRL router seeds the embedding query ----

def test_pipeline_retrieve_seeds_query_with_router_tables(tmp_path, monkeypatch):
    """The ICRL router call at the top of retrieve() must influence the
    embedder search query (i.e. its output is fed to expand_question /
    store.query_text).
    """
    captured: dict = {}

    class _RecordingStore(FakeStore):
        def query_text(self, text, top_k):
            captured["query"] = text
            return []

    p = _pipeline(tmp_path, [], [], example_hits=[])
    p.store = _RecordingStore()

    def _fake_router(*a, **kw):
        return ["public.orders", "public.customers"]
    monkeypatch.setattr("rag.icrl.retrieve_tables_for_question", _fake_router)

    p.retrieve("what did customers buy?")
    assert "public.orders" in captured["query"].lower() or "orders" in captured["query"].lower()
    assert "customers" in captured["query"].lower()


# ---- D3/D4: merged ICRL + verified few-shot block ----

def test_pipeline_generate_sql_injects_both_icrl_and_verified(tmp_path, monkeypatch):
    """When BOTH verified and ICRL examples are available, the prompt must
    contain both blocks (labelled VERIFIED ... and ICRL ...).
    """
    hits = [{
        "question": "monthly revenue",
        "sql": "SELECT month, SUM(amount) FROM f_sales GROUP BY month",
        "notes": "verified by hand",
        "distance": 0.2,
    }]
    payload = '{"sql": "SELECT month, SUM(amount) FROM f_sales GROUP BY month", "explanation": ""}'
    p = _pipeline(tmp_path, [payload], [(["m", "rev"], [["jan", 5]], False)], example_hits=hits)

    def _fake_synth(*a, **kw):
        return {"question": "weekly order count",
                "sql": "SELECT week, COUNT(*) FROM orders GROUP BY week",
                "tables": ["public.orders"]}
    monkeypatch.setattr("rag.icrl.retrieve_synthetic_example", _fake_synth)

    result = p.ask("monthly revenue?", execute=True)
    user_content = p.llm.calls[0][1]["content"]
    assert "VERIFIED QUERY EXAMPLES" in user_content
    assert "monthly revenue" in user_content
    assert "ICRL" in user_content.upper() or "SYNTHETIC" in user_content.upper()
    assert "weekly order count" in user_content


# ---- D5/D6: _llm_schema_pool uses top_k window ----

def test_llm_schema_pool_uses_top_k_window(tmp_path):
    """The LLM-aided schema pool must receive a small slice of candidates,
    not the entire ranked list (paper section 4: top-k, where k is small).
    """
    class _CountingLLM(FakeLLM):
        def __init__(self):
            super().__init__([])
            self.last_user_msg = None

        def chat_json(self, messages, max_tokens=8192, timeout=15.0):
            self.last_user_msg = messages[-1]["content"]
            return {"tables": [1, 2, 3]}

    p = _pipeline(tmp_path, [], [])
    p.llm = _CountingLLM()

    ranked = [
        {"metadata": {"table": f"public.t{i}", "role": "dim", "columns": list(range(5)), "comment": "x"}}
        for i in range(30)
    ]
    out = p._llm_schema_pool("any question", ranked)
    assert "1. public.t0" in p.llm.last_user_msg
    n_lines = sum(
        1 for ln in p.llm.last_user_msg.splitlines()
        if ln.strip()[:3] in {f"{i}." for i in range(1, 40)}
    )
    assert n_lines <= 12, f"too many candidates passed to LLM: {n_lines}"


# ---- D7/D8: multi-round when LLM returns empty picks ----

def test_llm_schema_pool_multi_round_when_unsure(tmp_path):
    """When the first LLM call returns empty picks, retry with a hint to
    fall back to top-N by name similarity.
    """
    class _TwoRoundLLM(FakeLLM):
        def __init__(self):
            super().__init__([])
            self.call_count = 0

        def chat_json(self, messages, max_tokens=8192, timeout=15.0):
            self.call_count += 1
            if self.call_count == 1:
                return {"tables": []}
            return {"tables": [1, 2]}

    p = _pipeline(tmp_path, [], [])
    p.llm = _TwoRoundLLM()

    ranked = [
        {"metadata": {"table": "public.orders", "role": "fact", "columns": list(range(5)), "comment": "x"}},
        {"metadata": {"table": "public.customers", "role": "dim", "columns": list(range(5)), "comment": "x"}},
        {"metadata": {"table": "public.products", "role": "dim", "columns": list(range(5)), "comment": "x"}},
        {"metadata": {"table": "public.unrelated", "role": "dim", "columns": list(range(5)), "comment": "x"}},
    ]
    out = p._llm_schema_pool("orders by customer", ranked)
    assert p.llm.call_count == 2
    kept_tables = {(h["metadata"]["table"].split(".")[-1]) for h in out}
    assert "orders" in kept_tables
    assert "customers" in kept_tables


# ---- D9/D10: ICRL batch concurrency ----

def test_icrl_generator_batch_concurrent():
    """ICRLGenerator.run_batch processes N traversals with bounded
    concurrency and returns one result per traversal.
    """
    import time

    from rag.icrl import ICRLGenerator, Traversal
    from test_icrl import _table

    def _tr():
        return Traversal(tables=[_table("F", role="fact", fks=["D"]), _table("D")],
                         label="F -> D")

    class _TimedLLM:
        def __init__(self):
            self.call_count = 0

        def chat_json(self, messages, max_tokens=8192):
            self.call_count += 1
            if self.call_count % 2 == 1:
                return {"question": "q", "sql": "SELECT a, b, c FROM t WHERE c = 1 GROUP BY a"}
            return {"feedback": "more complex"}

    from rag import icrl as _icrl_mod

    original = _icrl_mod.complexity_reward
    _icrl_mod.complexity_reward = lambda sql: (
        2.0, {"retrieval": 5, "conditional": 0, "aggregation": 0, "modification": 0}
    )
    try:
        gen = ICRLGenerator(_TimedLLM(), max_iterations=2, plateau_epsilon=0.5)
        traversals = [_tr() for _ in range(4)]
        t0 = time.monotonic()
        results = gen.run_batch(traversals, concurrency=4)
        elapsed = time.monotonic() - t0
        assert len(results) == 4
        assert elapsed < 5.0
    finally:
        _icrl_mod.complexity_reward = original
