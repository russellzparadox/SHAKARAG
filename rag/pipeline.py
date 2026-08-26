from __future__ import annotations

import json
import logging
import re
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
- RECENT CONVERSATION: if the question contains pronouns or ellipsis ("them", "their names", "those", "also show..."), resolve them against the most recent USER and ASSISTANT turns — the entity discussed there (e.g. suppliers) is what the user means. Do not switch to a different entity unless the current question names one explicitly.
- REPORTS & MULTI-TABLE ANALYSIS: for business reports (breakdowns, per-period trends, per-entity summaries), prefer CTEs (WITH clauses): stage filtered base data first, then aggregate. Join facts to dimensions strictly via the JOIN MAP conditions. Include ALL requested dimensions as GROUP BY columns, order results meaningfully (by period, then by measure DESC), and pick human-readable label columns from dimensions alongside IDs.
__HINTS__
- __DW_BLOCK__"""

SYSTEM_ANSWER = """You answer questions about a __ENGINE__ database using the result rows of a query that was executed against it.
Be concise and factual; cite concrete numbers from the results.
Structure your answer as:
1. A direct answer to the question (1-3 short paragraphs or bullets).
2. A line starting with "Query used:" summarizing in words what was queried (do not paste SQL).
3. A line starting with "Notes:" listing caveats (approximations like row estimates, archived-record assumptions, truncated results, related tables worth checking next).
If there are no rows, say so plainly and suggest what to check or query next.
IMPORTANT: Describe ONLY the data in the RESULT section of this request. RECENT CONVERSATION is context for understanding the question — never quote entities or rows from it; if the results contradict earlier turns, report the current results."""


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
    route: str = "sql"          # sql | document | chat
    doc_sources: list[str] = field(default_factory=list)
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
- the question is a follow-up whose pronouns ("their codes", "them", "those") resolve from RECENT CONVERSATION - just resolve and query
- the answer would be a subset of what they asked anyway ("which fields?", "everything?" is never real ambiguity)
- one candidate table clearly matches the entity mentioned
- the missing detail is visible in the schema context

Options must name concrete alternative interpretations (table or measure names), never vague words."""


FOLLOWUP_PATTERN = re.compile(
    r"\b(them|their|they|it|its|those|these|that one|the same|also|too|again|"
    r"same (ones?|table)|more (of|detail)|show (me )?(also|them))\b",
    re.IGNORECASE,
)


REPORT_INTENT = re.compile(
    r"\b(report|summary|breakdown|break down|per month|monthly|per week|weekly|per supplier|"
    r"per customer|per category|per product|by month|by year|by supplier|by customer|"
    r"trend|comparison|compare|over time|dashboard|kpi|top \d+|business review)\b",
    re.IGNORECASE,
)


def _is_followup(question: str) -> bool:
    """Heuristic: the question refers back to prior turns instead of naming an entity."""
    return bool(FOLLOWUP_PATTERN.search(question or ""))


def _is_report_request(question: str) -> bool:
    return bool(REPORT_INTENT.search(question or ""))


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

        # Contextual retrieval: for pronoun/ellipsis questions ("show their codes"),
        # blend entity words from the last user turns into the embedding query.
        search_query = question
        if getattr(self, "_history", None) and _is_followup(question):
            recent_user = [m["content"] for m in self._history if m["role"] == "user"][-2:]
            search_query = question + " " + " ".join(recent_user)
            logging.getLogger("chat.ask").info(
                "follow-up detected — contextual retrieval query: %s", search_query
            )

        expanded = expand_question(search_query, vocab_tokens) if vocab_tokens else search_query
        if expanded != search_query:
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
            if table and table not in seen_tables:
                seen_tables.add(table)
                tables_order.append(table)

        for hit in ordered:
            text = hit["text"]
            if used + len(text) > budget:
                continue
            parts.append(text)
            used += len(text)

        expanded_parts, expanded_tables = self._expand_join_neighbors(
            parts, tables_order, budget - used
        )
        if expanded_parts:
            parts = expanded_parts + parts
            used += sum(len(x) for x in expanded_parts)
            for t in reversed(expanded_tables):
                if t not in seen_tables:
                    tables_order.insert(0, t)

        context = "\n\n---\n\n".join(parts)
        join_map = self._join_map(tables_order, context)
        if join_map:
            context = join_map + "\n\n" + context
        return context, tables_order

    def _expand_join_neighbors(
        self, parts: list[str], tables: list[str], remaining_budget: int
    ) -> tuple[list[str], list[str]]:
        """1-hop FK expansion: pull schema chunks of referenced tables that retrieval
        missed, so multi-table questions have complete join targets. Budget-capped."""
        if remaining_budget <= 200:
            return [], []
        fk_re = re.compile(r"\[FK->([A-Za-z_][A-Za-z0-9_]*)\(")
        joined_text = "\n".join(parts)
        referenced: set[str] = set()
        for m in fk_re.finditer(joined_text):
            referenced.add(m.group(1))
        have_bare = {t.split(".")[-1] for t in tables}
        missing = sorted(referenced - have_bare)[:4]  # cap expansion
        new_parts: list[str] = []
        new_tables: list[str] = []
        added_budget = 0
        per_table_cap = max(remaining_budget // max(len(missing), 1), 0) if missing else 0
        try:
            collection = self.store.collection
        except Exception:
            return [], []
        for table_name in missing:
            if added_budget >= remaining_budget:
                break
            try:
                got = collection.get(
                    where={"table": table_name},
                    include=["documents", "metadatas"],
                    limit=2,
                )
            except Exception:
                continue
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            # prefer the stats/overview chunk (first one) — smallest useful summary
            for doc, meta in zip(docs, metas):
                if len(doc) > per_table_cap and new_parts:
                    continue
                qualified = (
                    f"{(meta or {}).get('schema', '')}.{table_name}"
                    if (meta or {}).get("schema")
                    else table_name
                )
                new_parts.append(doc)
                if qualified not in new_tables:
                    new_tables.append(qualified)
                added_budget += len(doc)
                break
        return new_parts, new_tables

    def _join_map(self, tables: list[str], context: str) -> str:
        """Derive explicit JOIN conditions between retrieved tables from FK annotations
        in the chunk texts, so multi-table questions get a ready join plan."""
        if len(tables) < 2:
            return ""

        # both qualified ("BI.DimX") and bare names must resolve
        by_bare = {}
        for t in tables:
            by_bare[t.split(".")[-1]] = t

        fk_re = re.compile(r"\[FK->([A-Za-z_][A-Za-z0-9_]*)\(([A-Za-z0-9_, ]+)\)\]")
        header_re = re.compile(
            r"^\s*([A-Za-z_][\w.]*\.[A-Za-z_][A-Za-z0-9_]*)\s+\(?(table|view)"
        )
        col_re = re.compile(r"^\s*-\s*\[?(\w+)")

        lines: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        current_table = None
        for raw_line in context.splitlines():
            tm = header_re.match(raw_line)
            if tm:
                full = tm.group(1)
                current_table = by_bare.get(full.split(".")[-1], full)
                continue
            if current_table is None or "-" not in raw_line[:3]:
                continue
            cm = col_re.match(raw_line)
            if not cm:
                continue
            col = cm.group(1)
            for m in fk_re.finditer(raw_line):
                ref_table, ref_cols = m.group(1), m.group(2).strip()
                ref_full = by_bare.get(ref_table)
                if not ref_full or current_table not in by_bare:
                    continue
                key = (current_table, col, ref_table)
                if key in seen:
                    continue
                seen.add(key)
                rc = ", ".join(c.strip() for c in ref_cols.split(","))
                lines.append(f"- {current_table}.{col} = {ref_full}.{rc}")

        if not lines:
            return ""
        return (
            "JOIN MAP (verified foreign-key relationships between the retrieved tables — "
            "use these exact conditions):\n" + "\n".join(lines[:25])
        )

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

    def _history_block(self) -> str:
        """Render recent conversation turns for context continuity."""
        if not getattr(self, "_history", None):
            return ""
        lines = ["RECENT CONVERSATION (for reference; the current request is the LAST message):"]
        for m in self._history[-8:]:
            role = "USER" if m["role"] == "user" else "ASSISTANT"
            lines.append(f"{role}: {m['content'][:600]}")
        return "\n".join(lines) + "\n\n"

    # ---- intent routing -------------------------------------------------

    def _doc_hits(self, hits: list[dict]) -> list[dict]:
        return [h for h in hits if (h.get("metadata") or {}).get("kind") == "document"]

    def _looks_conversational(self, question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return True
        if re.fullmatch(r"[\s hi hello hey سلام درود ?!.,،؟]+", q):
            return True
        if re.search(
            r"^(hi|hello|hey|سلام|درود|thanks|merci|ممنون|چه خبر|خوبی)\b", q
        ) and len(q.split()) <= 6:
            return True
        return False

    def _route_intent(self, question: str, hits: list[dict], context: str) -> str:
        """Decide between sql / document / chat.

        Heuristics first; LLM arbitration when ambiguous.
        """
        doc_hits = self._doc_hits(hits)
        schema_hits = [h for h in hits if (h.get("metadata") or {}).get("kind") != "document"]
        has_docs = bool(doc_hits)
        has_schema = bool(schema_hits)

        # Conversational messages always win, regardless of what retrieval found;
        # empty retrieval alone does NOT force chat (the schema index may have missed).
        if self._looks_conversational(question):
            return "chat"
        if not hits:
            return "sql"

        # Doc-only retrieval is not proof by itself: data-flavored questions
        # (count/sum/list/how many) should stay on SQL even when a document mentions
        # the entity. Arbitrate when both signals could apply.
        if has_docs and self.llm is None:
            return "document"

        if self.llm is None:
            return "sql"
        if has_docs:
            # mixed or doc-only retrieval with an LLM available — arbitrate with evidence
            return self._arbitrate_route(question, doc_hits, schema_hits)

        q = question.strip()
        if self._looks_conversational(q):
            return "chat"

        return "sql"

    def _arbitrate_route(
        self, question: str, doc_hits: list[dict], schema_hits: list[dict]
    ) -> str:
        try:
            parsed = self.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You route user questions. Answer ONLY JSON:\n"
                            '{"route": "sql" | "document"}\n'
                            '"sql" = the question asks about data records, counts, '
                            "aggregates, filtering rows — answerable by querying tables. "
                            "Words like how many, total, quantity, list, show strongly "
                            "indicate sql even if a document mentions the entity.\n"
                            '"document" = the question asks about content of uploaded '
                            "documents/reports/contracts/notes — terms, descriptions, "
                            "explanations, summaries."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "QUESTION: "
                        + question
                        + "\n\nDOCUMENT evidence found:\n"
                        + "\n".join(
                            f"- {(h['metadata'] or {}).get('source','?')}: "
                            + h["text"][:150]
                            for h in doc_hits[:3]
                        )
                        + "\n\nSCHEMA evidence found:\n"
                        + "\n".join(
                            "- " + ((h["metadata"] or {}).get("table", "?"))
                            for h in schema_hits[:5]
                        ),
                    },
                ],
                max_tokens=40,
            )
        except LLMError:
            return "sql"
        route = (parsed or {}).get("route")
        return route if route in ("sql", "document") else "sql"

    def _answer_from_documents(
        self, question: str, hits: list[dict], result: RagResult, answer_language: str
    ) -> RagResult:
        """Answer from document chunks retrieved for the question."""
        doc_hits = self._doc_hits(hits)
        sources: list[str] = []
        blocks: list[str] = []
        used = 0
        budget = max(self.settings.context_char_budget // 2, 4000)
        for h in doc_hits:
            m = h["metadata"] or {}
            src = m.get("source", "?")
            sec = m.get("section") or ""
            if src not in sources:
                sources.append(src)
            block = f"[{src}" + (f" — {sec}]" if sec else "]") + f"\n{h['text']}"
            if used + len(block) > budget:
                continue
            blocks.append(block)
            used += len(block)

        result.doc_sources = sources

        system = (
            "You answer questions using ONLY the provided DOCUMENT excerpts.\n"
            "Cite sources inline like [filename]. If the excerpts don't contain the answer, "
            "say so plainly and suggest what to check. Do not invent facts."
        )
        lang_name = ANSWER_LANGUAGES.get((answer_language or "auto").lower())
        if lang_name and answer_language != "auto":
            system += (
                f"\n\nIMPORTANT: Write your ENTIRE answer in {lang_name}. "
                "Keep file names unchanged."
            )

        user = (
            f"{self._history_block()}QUESTION: {question}\n\n"
            "DOCUMENT EXCERPTS:\n\n" + "\n\n---\n\n".join(blocks) + "\n\nAnswer now."
        )
        result.answer = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        result.tables_used = sorted({(h["metadata"] or {}).get("source", "?") for h in doc_hits})
        return result

    def _conversational_answer(
        self, question: str, tables: list[str], result: RagResult, answer_language: str
    ) -> RagResult:
        """Greetings / small talk / capability questions — no SQL."""
        system = (
            "You are ShakaRAG, an assistant connected to company databases and documents. "
            "The user sent a conversational message (greeting, thanks, or a question about "
            "your abilities). Reply briefly, warmly, in one short paragraph. If relevant, "
            "mention what they can ask: counts, lists, trends from the database, or search "
            "their uploaded documents. Do NOT write SQL. Do NOT invent data."
        )
        lang_name = ANSWER_LANGUAGES.get((answer_language or "auto").lower())
        if lang_name and answer_language != "auto":
            system += f"\n\nIMPORTANT: Write your ENTIRE reply in {lang_name}."
        ctx_note = ""
        if tables:
            ctx_note = (
                "\n\nConnected database has tables including: "
                + ", ".join(tables[:10])
                + "."
            )
        user = f"{self._history_block()}MESSAGE: {question}{ctx_note}\n\nReply now."
        result.answer = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return result

    def _data_preview_block(self, tables: list[str], context: str) -> str:
        """Sample real rows from the top candidate tables to ground the SQL generation.

        Best-effort only; disabled via DATA_PREVIEW=0.
        """
        if not getattr(self.settings, "data_preview", True):
            return ""
        preview_lines: list[str] = []
        seen: set[str] = set()
        # resolve bare table names to their schema-qualified form when possible
        schema_by_bare: dict[str, str] = {}
        try:
            for meta in self.store.all_table_metas():
                bare = (meta.get("table") or "").split(".")[-1]
                if meta.get("schema") and bare:
                    schema_by_bare[bare] = f'{meta["schema"]}.{bare}'
        except Exception:
            pass
        is_mssql = hasattr(self.dialect, "label") and "SQL Server" in self.dialect.label
        for t in tables[:4]:
            if t in seen:
                continue
            seen.add(t)
            try:
                qualified = t if "." in t else schema_by_bare.get(t.split(".")[-1], t)
                if is_mssql:
                    sql = f"SELECT TOP 3 * FROM {qualified}"
                else:
                    sql = f"SELECT * FROM {qualified} LIMIT 3"
                cols, rows, _trunc = self.run_sql(sql)
                if not rows:
                    continue
                shown = [
                    " | ".join(str(v)[:30] if v is not None else "NULL" for v in r)
                    for r in rows
                ]
                header = " | ".join(cols[:8])
                body = "\n".join(shown[:3])
                preview_lines.append(f"SAMPLE ROWS FROM {t}:\n{header}\n{body}")
                if len(preview_lines) >= 3:
                    break
            except Exception:
                continue  # preview is best-effort only
        if not preview_lines:
            return ""
        return (
            "\nREAL DATA PREVIEW (actual current values — match their exact spelling/format):\n"
            + "\n\n".join(preview_lines)
            + "\n"
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
                "content": f"CONTEXT:\n{context}\n\n{self._history_block()}QUESTION: {question}"
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
            {"role": "user", "content": f"CONTEXT:\n{context}\n\n{self._history_block()}QUESTION: {question}"},
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
            f"QUESTION: {question}\n\n{self._history_block()}EXECUTED SQL:\n{sql}\n\n"
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
                        + f"\n\n{self._history_block()}QUESTION: {question}",
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
        history: list[dict[str, str]] | None = None,
    ) -> RagResult:
        result = RagResult(question=question)
        self._history: list[dict[str, str]] = [
            m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        hits = self.retrieve(question, top_k=top_k)
        context, tables = self.build_context(hits)
        result.tables_used = tables
        result.context = context

        if dry_run:
            return result

        # ---- intent routing: documents / chat / sql ------------------------
        route = self._route_intent(question, hits, context)
        result.route = route
        logging.getLogger("chat.ask").info("route=%s", route)

        if route == "document":
            return self._answer_from_documents(question, hits, result, answer_language)
        if route == "chat":
            return self._conversational_answer(
                question, tables, result, answer_language
            )
        # else: fall through to SQL path (with data preview enrichment)

        if clarify and not self._few_shot_block(question):
            decision = self._clarify_decision(question, hits)
            if decision:
                result.needs_clarification = True
                result.clarify_question = decision["question"]
                result.options = decision["options"]
                return result

        gen = self.generate_sql(
            question, context + self._data_preview_block(tables, context)
        )
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
