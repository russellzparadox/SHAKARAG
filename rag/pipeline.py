from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

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
__HINTS__
- __DW_BLOCK__"""

SYSTEM_ANSWER = """You answer questions about a __ENGINE__ database using the result rows of a query that was executed against it.
Be concise and factual; cite concrete numbers from the results.
Structure your answer as:
1. A direct answer to the question (1-3 short paragraphs or bullets).
2. A line starting with "Query used:" summarizing in words what was queried (do not paste SQL).
3. A line starting with "Notes:" listing caveats (approximations like row estimates, archived-record assumptions, truncated results, related tables worth checking next).
If there are no rows, say so plainly and suggest what to check or query next."""


ANSWER_LANGUAGES = {
    "auto": "",
    "en": "English",
    "fa": "Persian (Farsi / فارسی)",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "tr": "Turkish",
    "zh": "Simplified Chinese",
    "ru": "Russian",
}


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
    needs_clarification: bool = False
    clarify_question: str | None = None
    options: list[str] = field(default_factory=list)


SYSTEM_CLARIFY = """You are a careful analytics gatekeeper standing between the user and a database.
Given the user's QUESTION and the most relevant candidate tables found in the schema, decide if the request can be queried confidently.

Output ONLY JSON, one of:
{"action": "proceed"}
{"action": "clarify", "question": "<ONE short clarifying question>", "options": ["<option 1>", "<option 2>", "<option 3>"]}

Default to {"action": "proceed"}. When in doubt, DO NOT ask.

Ask ONLY when two or more genuinely different INTERPRETATIONS exist and picking wrong would produce a materially different answer - for example the entity name matches several unrelated tables (customer vs supplier), or a shared column name means totally different measures.

NEVER ask when:
- the user already named what they want ("supplier info of Xperial", "sales per month") - just query it
- the answer would be a subset of what they asked anyway ("which fields?", "everything?" is never real ambiguity)
- one candidate table clearly matches the entity mentioned
- the missing detail is visible in the schema context

Options must name concrete alternative interpretations (table or measure names), never vague words."""


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
        from .rerank import _name_tokens_for, expand_question, rerank_hits

        k = top_k or self.settings.top_k
        try:
            vocab_tokens: set[str] = set()
            for meta in self.store.all_table_metas():
                vocab_tokens |= _name_tokens_for({"metadata": meta})
        except Exception:
            vocab_tokens = set()
        expanded = expand_question(question, vocab_tokens) if vocab_tokens else question
        if expanded != question:
            logging.getLogger("chat.ask").info(
                "query expanded for schema vocabulary: %s", expanded
            )
        hits = self.store.query_text(expanded, top_k=max(k * 3, 12))
        return rerank_hits(hits, question)[:k]

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

    def _example_store(self):
        if getattr(self, "_examples", None) is None:
            from .examples import ExampleStore

            self._examples = ExampleStore(
                str(self.settings.chroma_dir), self.settings.collection, self.store.embedder
            )
        return self._examples

    def _few_shot_block(self, question: str, k: int | None = None) -> str:
        try:
            hits = self._example_store().search(question, k=k or 2)
        except Exception:
            return ""
        good = [h for h in hits if h["distance"] < 0.85]
        if not good:
            return ""
        import logging

        logging.getLogger("chat.ask").info("using %d verified example(s)", len(good))
        parts = []
        for h in good:
            note = f"\nNotes: {h['notes']}" if h["notes"] else ""
            parts.append(f"---\nQ: {h['question']}\nSQL:\n{h['sql']}{note}")
        return (
            "\n\nVERIFIED QUERY EXAMPLES previously confirmed correct for this database "
            "(imitate their join paths, table usage and value formatting):\n" + "\n".join(parts)
        )

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
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"
                + self._few_shot_block(question),
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

    def _repair_sql(self, question: str, context: str, bad_sql: str, error: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": render_sql_system(
                    self.dialect.label, self.dialect.prompt_hints(), self.settings.max_rows
                ),
            },
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
            {"role": "assistant", "content": json.dumps({"sql": bad_sql})},
            {
                "role": "user",
                "content": f"Executing that query raised this database error:\n{error}\n\n"
                "Rewrite the corrected read-only query. Respond with ONLY the JSON object.",
            },
        ]
        parsed = self.llm.chat_json(messages)
        sql = (parsed or {}).get("sql")
        if not sql:
            raise SQLGuardError("Repair attempt returned no SQL.")
        return {"sql": validate_sql(sql), "explanation": (parsed or {}).get("explanation")}

    def _name_tokens_for(hit: dict) -> set[str]:
        from .rerank import _name_tokens_for

        return _name_tokens_for(hit)

    def _table_inventory(self) -> list[str]:
        lines: list[str] = []
        try:
            for m in self.store.all_table_metas():
                role = f" [{m.get('wh_role')}]" if m.get("wh_role") else ""
                label = f" — {m['label']}" if m.get("label") else ""
                lines.append(
                    f"- {m.get('schema','')}.{m.get('table','')}{role}{label}"
                )
        except Exception:
            pass
        return lines[:250] or ["(inventory unavailable)"]

    def _retry_empty(self, question: str, context: str, empty_sql: str):
        logging.getLogger("chat.ask").info(
            "empty result — asking model to reconsider (tables tried elsewhere)"
        )
        messages = [
            {
                "role": "system",
                "content": render_sql_system(
                    self.dialect.label, self.dialect.prompt_hints(), self.settings.max_rows
                ),
            },
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
            {"role": "assistant", "content": json.dumps({"sql": empty_sql})},
            {
                "role": "user",
                "content": "That query executed fine but returned ZERO rows. The entity the user "
                "named may live in a DIFFERENT table, or in another human-readable column, or "
                "under a different spelling.\n"
                "Here is EVERY table in this database — pick the one whose name truly matches "
                "the entity the user mentioned:\n"
                + "\n".join(self._table_inventory())
                + "\n\nRewrite the query against that table/column using wildcard matching on a "
                'word stem. If nothing could contain it, return {"sql": "'
                + empty_sql.replace('"', "")
                + '", "explanation": "no better candidate"}. '
                "Respond with ONLY the JSON object.",
            },
        ]
        parsed = self.llm.chat_json(messages)
        sql = (parsed or {}).get("sql") if isinstance(parsed, dict) else None
        if not sql:
            return None
        validated = validate_sql(sql)
        if not validated or validated.strip() == empty_sql.strip():
            return None
        explanation = (parsed or {}).get("explanation") or ""
        if "no better candidate" in explanation.lower():
            return None
        return {"sql": validated, "explanation": explanation}

    def run_sql(self, sql: str) -> tuple[list[str], list[list], bool]:
        return self.dialect.execute_readonly(sql, self.settings.max_rows)

    def synthesize_answer(self, question: str, sql: str, columns: list[str], rows: list[list], truncated: bool, answer_language: str = "auto") -> str:
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
        system = render_answer_system(self.dialect.label)
        language_name = ANSWER_LANGUAGES.get((answer_language or "auto").lower())
        if language_name and answer_language != "auto":
            system += (
                f"\n\nIMPORTANT: Write your ENTIRE answer in {language_name}, including the "
                "'Query used' and 'Notes' sections. Keep SQL keywords, table names and column "
                "names unchanged. Use proper locale conventions for numbers and dates."
            )
        return self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    def _clarify_decision(self, question: str, hits: list[dict]) -> dict | None:
        if self.llm is None or not hits:
            return None
        lines = []
        for h in hits[:12]:
            m = h.get("metadata") or {}
            label = f" — {m['label']}" if m.get("label") else ""
            role = f" [{m['wh_role']}]" if m.get("wh_role") else ""
            lines.append(f"- {m.get('schema','')}.{m.get('table','')}{label}{role}")
        try:
            parsed = self.llm.chat_json(
                [
                    {"role": "system", "content": SYSTEM_CLARIFY},
                    {
                        "role": "user",
                        "content": "CANDIDATE TABLES:\n" + "\n".join(lines)
                        + f"\n\nQUESTION: {question}",
                    },
                ],
                max_tokens=400,
            )
        except LLMError:
            return None
        if isinstance(parsed, dict) and parsed.get("action") == "clarify" and parsed.get("question"):
            options = [str(o) for o in (parsed.get("options") or [])][:4]
            return {"question": str(parsed["question"]), "options": options}
        return None

    def ask(
        self,
        question: str,
        execute: bool = True,
        dry_run: bool = False,
        top_k: int | None = None,
        answer_language: str = "auto",
        clarify: bool = False,
    ) -> RagResult:
        result = RagResult(question=question)
        hits = self.retrieve(question, top_k=top_k)
        context, tables = self.build_context(hits)
        result.tables_used = tables
        result.context = context

        if dry_run:
            return result

        if clarify and not self._few_shot_block(question):
            decision = self._clarify_decision(question, hits)
            if decision:
                result.needs_clarification = True
                result.clarify_question = decision["question"]
                result.options = decision["options"]
                return result

        gen = self.generate_sql(question, context)
        result.sql = gen["sql"]
        result.explanation = gen.get("explanation")

        if not execute:
            return result

        max_exec_attempts = 2
        for attempt in range(1, max_exec_attempts + 1):
            try:
                columns, rows, truncated = self.run_sql(result.sql)
                if rows:
                    break
                if attempt == max_exec_attempts or self.llm is None or not columns:
                    break
                try:
                    alt = self._retry_empty(question, context, result.sql)
                except Exception:
                    break
                if not alt:
                    break
                result.sql = alt["sql"]
                result.explanation = alt.get("explanation") or result.explanation
            except Exception as exc:
                if attempt == max_exec_attempts or self.llm is None:
                    result.error = f"SQL execution failed: {exc}"
                    return result
                try:
                    fixed = self._repair_sql(question, context, result.sql, str(exc))
                except Exception:
                    result.error = f"SQL execution failed: {exc}"
                    return result
                result.sql = fixed["sql"]
                result.explanation = fixed.get("explanation") or result.explanation

        result.columns = columns
        result.rows = rows
        result.row_count = len(rows)
        result.truncated = truncated

        try:
            result.answer = self.synthesize_answer(
                question, result.sql, columns, rows, truncated, answer_language=answer_language
            )
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
    from .warehouse import warehouse_hints_block

    return (
        SYSTEM_SQL.replace("__ENGINE__", label)
        .replace("__MAX_ROWS__", str(max_rows))
        .replace("__HINTS__", hints)
        .replace("- __DW_BLOCK__", warehouse_hints_block())
    )


def render_answer_system(label: str) -> str:
    return SYSTEM_ANSWER.replace("__ENGINE__", label)


def make_pipeline(settings: Settings) -> RagPipeline:
    from .dialects import get_dialect
    from .embeddings import get_embedder
    from .store import VectorStore as VS

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
