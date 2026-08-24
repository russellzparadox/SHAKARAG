from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .config import Settings
from .llm import LLMClient, LLMError
from .sqlguard import SQLGuardError, validate_sql
from .store import VectorStore

SYSTEM_SQL = """You are a senior __ENGINE__ analyst.
Given retrieved schema context (tables, columns, descriptions, relations) and a natural-language question, write ONE read-only query.

Rules:
- Output ONLY JSON: {"sql": "...", "explanation": "short why/how"}
- The SQL must be a single statement that only reads data. Never write INSERT/UPDATE/DELETE/DDL or any write/maintenance command.
- Use ONLY tables and columns that appear in the CONTEXT. Copy names exactly.
- Join via the foreign keys shown in the context. Many2many/join tables are listed as relationships.
- Odoo specifics (if the schema is an Odoo database): archived records have active = false - filter active = true when the user wants current/live records. States/codes are usually exact lowercase strings; human text is best matched case-insensitively.
- For counts/aggregates use GROUP BY and ORDER BY the aggregate DESC when ranking.
- Add a row LIMIT (<= __MAX_ROWS__) for row-level listings unless the user asks for totals.
- If the CONTEXT does not contain what you need, still produce your best-effort query from related tables you do see and explain the gap in "explanation".
__HINTS__"""

SYSTEM_ANSWER = """You answer questions about a __ENGINE__ database using the result rows of a query that was executed against it.
Be concise and factual; cite concrete numbers from the results.
Structure your answer as:
1. A direct answer to the question (1-3 short paragraphs or bullets).
2. A line starting with "Query used:" summarizing in words what was queried (do not paste SQL).
3. A line starting with "Notes:" listing caveats (approximations like row estimates, archived-record assumptions, truncated results, related tables worth checking next).
If there are no rows, say so plainly and suggest what to check or query next."""


@dataclass
class RagResult:
    question: str
    sql: str | None = None
    explanation: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    tables_used: list[str] = field(default_factory=list)
    context: str | None = None
    answer: str | None = None
    error: str | None = None


class RagPipeline:
    def __init__(self, settings: Settings, store: VectorStore, llm: LLMClient | None, dialect=None) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        if dialect is None:
            from .dialects import get_dialect

            dialect = get_dialect(settings)
        self.dialect = dialect

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.settings.top_k
        return self.store.query_text(question, top_k=k)

    def build_context(self, hits: list[dict]) -> tuple[str, list[str]]:
        budget = self.settings.context_char_budget
        seen_tables: set[str] = set()
        tables_order: list[str] = []
        parts: list[str] = []
        used = 0

        ordered = sorted(hits, key=lambda h: h["distance"])
        for hit in ordered:
            table = (hit.get("metadata") or {}).get("table", "")
            if table not in seen_tables:
                seen_tables.add(table)
                tables_order.append(table)

        for hit in ordered:
            text = hit["text"]
            if used + len(text) > budget:
                continue
            parts.append(text)
            used += len(text)

        context = "\n\n---\n\n".join(parts)
        return context, tables_order

    def generate_sql(self, question: str, context: str) -> dict:
        if self.llm is None:
            raise LLMError(
                "No LLM configured. Set LLM_BASE_URL and LLM_MODEL "
                "(any OpenAI-compatible endpoint, e.g. Ollama at http://localhost:11434/v1)."
            )
        messages = [
            {
                "role": "system",
                "content": render_sql_system(
                    self.dialect.label, self.dialect.prompt_hints(), self.settings.max_rows
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            },
        ]
        parsed = self.llm.chat_json(messages)

        attempts = 0
        while attempts < 2:
            sql = parsed.get("sql") if isinstance(parsed, dict) else None
            try:
                if not sql:
                    raise SQLGuardError("Missing 'sql' field.")
                validate_sql(sql)
                break
            except SQLGuardError as exc:
                attempts += 1
                messages.append({"role": "assistant", "content": str(parsed)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your SQL was rejected by the safety validator: {exc}. "
                        "Fix it and respond again with ONLY the JSON object.",
                    }
                )
                parsed = self.llm.chat_json(messages)
        else:
            raise SQLGuardError("Model failed to produce valid read-only SQL after retries.")

        explanation = parsed.get("explanation") if isinstance(parsed, dict) else ""
        return {"sql": validate_sql(parsed["sql"]), "explanation": explanation}

    def run_sql(self, sql: str) -> tuple[list[str], list[list], bool]:
        return self.dialect.execute_readonly(sql, self.settings.max_rows)

    def synthesize_answer(self, question: str, sql: str, columns: list[str], rows: list[list], truncated: bool) -> str:
        if self.llm is None:
            raise LLMError("No LLM configured for answer synthesis.")

        max_show = 25
        shown = rows[:max_show]
        lines = ["| " + " | ".join(columns) + " |"]
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in shown:
            cells = [str(v)[:80] if v is not None else "NULL" for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        if len(rows) > max_show:
            lines.append(f"... ({len(rows) - max_show} more rows not shown)")
        if truncated:
            lines.append(f"(result capped at {self.settings.max_rows} rows)")

        user = (
            f"QUESTION: {question}\n\nEXECUTED SQL:\n{sql}\n\n"
            f"RESULT ({len(rows)} rows):\n{chr(10).join(lines)}\n\nAnswer now."
        )
        return self.llm.chat(
            [
                {"role": "system", "content": render_answer_system(self.dialect.label)},
                {"role": "user", "content": user},
            ]
        )

    def ask(
        self,
        question: str,
        execute: bool = True,
        dry_run: bool = False,
        top_k: int | None = None,
    ) -> RagResult:
        result = RagResult(question=question)
        hits = self.retrieve(question, top_k=top_k)
        context, tables = self.build_context(hits)
        result.tables_used = tables
        result.context = context

        if dry_run:
            return result

        gen = self.generate_sql(question, context)
        result.sql = gen["sql"]
        result.explanation = gen.get("explanation")

        if not execute:
            return result

        try:
            columns, rows, truncated = self.run_sql(result.sql)
        except Exception as exc:
            result.error = f"SQL execution failed: {exc}"
            return result

        result.columns = columns
        result.rows = rows
        result.row_count = len(rows)
        result.truncated = truncated

        try:
            result.answer = self.synthesize_answer(question, result.sql, columns, rows, truncated)
        except LLMError as exc:
            result.answer = None
            result.error = str(exc)

        return result


def format_markdown_table(columns: list[str], rows: list[list], limit: int = 25) -> str:
    if not columns:
        return "(no results)"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(str(v)[:60] if v is not None else "NULL" for v in row) + " |")
    if len(rows) > limit:
        out.append(f"... (+{len(rows) - limit} more rows)")
    return "\n".join(out)


def render_sql_system(label: str, hints: str, max_rows: int) -> str:
    return (
        SYSTEM_SQL.replace("__ENGINE__", label)
        .replace("__MAX_ROWS__", str(max_rows))
        .replace("__HINTS__", hints)
    )


def render_answer_system(label: str) -> str:
    return SYSTEM_ANSWER.replace("__ENGINE__", label)


def make_pipeline(settings: Settings) -> RagPipeline:
    from .embeddings import get_embedder
    from .store import VectorStore as VS
    from .dialects import get_dialect

    dialect = get_dialect(settings)
    embedder = get_embedder(settings)
    store = VS(str(settings.chroma_dir), settings.collection, embedder)
    llm = (
        LLMClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
        )
        if settings.llm_ready
        else None
    )
    return RagPipeline(settings=settings, store=store, llm=llm, dialect=dialect)
