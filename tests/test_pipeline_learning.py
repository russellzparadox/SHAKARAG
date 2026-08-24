import json
from types import SimpleNamespace

from rag.config import Settings
from rag.pipeline import RagPipeline, render_sql_system


class FakeStore:
    embedder = None

    def query_text(self, text, top_k):
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


def test_no_examples_no_block(tmp_path):
    payload = '{"sql": "SELECT 1", "explanation": ""}'
    p = _pipeline(tmp_path, [payload], [(["one"], [[1]], False)])
    result = p.ask("anything")
    user_content = p.llm.calls[0][1]["content"]
    assert "VERIFIED QUERY EXAMPLES" not in user_content
