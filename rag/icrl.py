"""ICRL-based synthetic QA generation for SHAKARAG.

Implements the core method from "In-Context Reinforcement Learning based
Retrieval-Augmented Generation for Text-to-SQL" (Toteja et al.):

1. Schema graph construction — tables as nodes, FKs as type-3 edges.
2. Graph traversals (bounded random walks) → serialized join contexts.
3. Synthetic NL question + SQL generation from a traversal by a base LLM.
4. ICRL loop: a reward function scores the SQL's complexity across four keyword
   buckets; a feedback LLM tells the base LLM how to make the question harder;
   iterate until reward plateaus or max iterations.
5. Final triplets (question, tables, SQL) are indexed into a dedicated Chroma
   collection used for table/schema routing at ask time, and can also serve as
   few-shot candidates.

Everything is prompt-based: no fine-tuning.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .introspect import TableRecord
from .llm import LLMClient, LLMError
from .icrl_graph import (  # re-exports — paper Alg. A.1 + A.2
    EdgeType,
    GraphEdge,
    SchemaGraph,
    TraversalPath,
    build_from_tables as build_schema_graph_typed,
    enumerate_traversals as enumerate_traversals_typed,
)

# Optional SQL parser. Used by `parse_sql` to validate executability of
# synthetic queries before they're scored. Gracefully degrades to no-op when
# sqlglot is not installed (we still keep the read-only check via the
# existing sqlguard).
try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import ParseError as _SqlglotParseError
    _HAS_SQLGLOT = True
except Exception:  # pragma: no cover
    _HAS_SQLGLOT = False
    _SqlglotParseError = Exception  # type: ignore[assignment,misc]


def parse_sql(sql: str, dialect: str = "") -> bool:
    """Return True iff `sql` parses as a single statement.

    Uses sqlglot when available; otherwise falls back to a non-validating
    acceptance (returns True). Errors raise SQLParseError.
    """
    if not _HAS_SQLGLOT:
        return True
    try:
        statements = sqlglot.parse(sql, read=dialect or None)
    except _SqlglotParseError as exc:
        raise SQLParseError(str(exc)) from exc
    if not statements or len(statements) != 1:
        raise SQLParseError(f"expected 1 statement, got {len(statements) if statements else 0}")
    return True


class SQLParseError(Exception):
    """Raised when sqlglot (or future parser) rejects a candidate query."""


logger = logging.getLogger("chat.icrl")

# ---------------------------------------------------------------------------
# Complexity reward (paper §A.3)
# ---------------------------------------------------------------------------

# Multi-word JOIN qualifiers — counted with a single higher-weight pattern
# and then SUBTRACTED from the bare-JOIN count, so a single SQL "INNER JOIN"
# contributes INNER_JOIN (weight 3) and NOT the bare-JOIN (weight 2) too.
_JOIN_QUALIFIERS = ("INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER")

BUCKETS: dict[str, dict[str, tuple[str, int]]] = {
    # B1 data retrieval & filtering
    "retrieval": {
        r"\bSELECT\b": 1, r"\bFROM\b": 1,
        r"\bINNER\s+JOIN\b": 3,
        r"\bLEFT\s+(?:OUTER\s+)?JOIN\b": 3,
        r"\bRIGHT\s+(?:OUTER\s+)?JOIN\b": 3,
        r"\bFULL\s+(?:OUTER\s+)?JOIN\b": 3,
        r"\bCROSS\s+JOIN\b": 2,
        r"\bJOIN\b": 2,  # bare JOIN (will be neutralised by qualifier matches below)
        r"\bON\b": 2, r"\bWHERE\b": 2, r"\bGROUP\s+BY\b": 3,
        r"\bHAVING\b": 3, r"\bORDER\s+BY\b": 2, r"\bDISTINCT\b": 2,
        r"\bLIMIT\b": 1, r"\bTOP\b": 1, r"\bOFFSET\b": 1,
    },
    # B2 data modification (penalised — read-only system)
    "modification": {
        r"\bINSERT\b": 2, r"\bUPDATE\b": 3, r"\bDELETE\b": 4,
        r"\bDROP\b": 5, r"\bALTER\b": 5, r"\bTRUNCATE\b": 4,
    },
    # B3 conditional logic
    "conditional": {
        r"\bAND\b": 1, r"\bOR\b": 1, r"\bNOT\b": 1,
        r"\bIN\s*\(": 2, r"\bBETWEEN\b": 2, r"\bLIKE\b": 2,
        r"\bIS\s+NOT\s+NULL\b": 3, r"\bIS\s+NULL\b": 2,
        r"\bCASE\s+WHEN\b": 3, r"\bWHEN\b": 2, r"\bTHEN\b": 2, r"\bELSE\b": 2,
        r"\bEXISTS\b": 3,
    },
    # B4 aggregation & sorting
    "aggregation": {
        r"\bAVG\s*\(": 3, r"\bSUM\s*\(": 3, r"\bCOUNT\s*\(": 3,
        r"\bMIN\s*\(": 3, r"\bMAX\s*\(": 3, r"\bCOALESCE\s*\(": 2,
        r"\bDATEPART\s*\(": 2, r"\bYEAR\s*\(": 2, r"\bMONTH\s*\(": 2,
        r"\bASC\b": 1, r"\bDESC\b": 1, r"\bCAST\s*\(": 2, r"\bCONVERT\s*\(": 2,
    },
}

NEGATIVE_REWARD = {"modification"}  # read-only system: DML keywords reduce reward

# compile once at import time
_BUCKET_PATTERNS: dict[str, dict[re.Pattern, int]] = {
    bucket: {re.compile(pat, re.IGNORECASE): w for pat, w in keywords.items()}
    for bucket, keywords in BUCKETS.items()
}

# pattern that finds each qualified JOIN occurrence (INNER JOIN, LEFT JOIN, …)
_QUALIFIED_JOIN_RE = re.compile(
    r"\b(?:" + "|".join(_JOIN_QUALIFIERS) + r")\s+(?:OUTER\s+)?JOIN\b",
    re.IGNORECASE,
)


def complexity_reward(sql: str) -> tuple[float, dict[str, int]]:
    """Paper eq. (1)+(2): weighted bucket-frequency reward. Returns (reward, counts)."""
    counts: dict[str, int] = {b: 0 for b in BUCKETS}
    for bucket, patterns in _BUCKET_PATTERNS.items():
        for pattern, weight in patterns.items():
            n = len(pattern.findall(sql))
            if n:
                counts[bucket] += n * weight

    # The "qualified" patterns (INNER JOIN, LEFT JOIN, …) double-count with
    # the bare-JOIN pattern. Subtract the bare-JOIN contribution for each
    # qualified match so the totals reflect the *intent* (one JOIN = weight
    # of its qualifier, not qualifier+2).
    if "retrieval" in counts:
        bare_join_w = _BUCKET_PATTERNS["retrieval"][re.compile(r"\bJOIN\b", re.IGNORECASE)]
        n_qualified = len(_QUALIFIED_JOIN_RE.findall(sql))
        if n_qualified:
            counts["retrieval"] -= n_qualified * bare_join_w

    # no negative values
    for b, v in counts.items():
        if v < 0:
            counts[b] = 0

    total_weighted = sum(counts.values())
    if total_weighted == 0:
        return 0.0, counts

    # normalise each bucket share then apply bucket weights:
    # retrieval & aggregation are desirable; conditional adds complexity;
    # modification is penalised (read-only system).
    weights = {"retrieval": 1.0, "conditional": 1.2, "aggregation": 1.5, "modification": -2.0}
    reward = sum(
        weights.get(b, 1.0) * (freq / total_weighted) * total_weighted / 10.0
        for b, freq in counts.items()
    )
    return round(reward, 3), counts


# Paper §A.3 operator-suggestion table: which SQL operators help close a
# given bucket gap. The coach LLM consults this when its previous question
# was too simple in a particular dimension.
OPERATOR_SUGGESTIONS: dict[str, list[str]] = {
    "retrieval": [
        "JOIN multiple tables (JOIN, LEFT JOIN, INNER JOIN)",
        "add ON ... = ... join conditions",
        "filter with WHERE on at least two columns",
    ],
    "conditional": [
        "use AND / OR to combine predicates",
        "add IN (...) or BETWEEN ... AND ...",
        "use CASE WHEN ... THEN ... ELSE ... END for derived columns",
        "use LIKE '%pattern%' for text search",
        "use IS NULL / IS NOT NULL",
    ],
    "aggregation": [
        "use GROUP BY with COUNT, SUM, AVG, MIN, or MAX",
        "add HAVING to filter aggregated groups",
        "ORDER BY <aggregate> ASC or DESC to rank results",
        "use COALESCE or CAST to coerce values",
    ],
    "modification": [
        # Should be empty: this bucket is penalised, not encouraged.
    ],
}


def operator_suggestions(bucket: str) -> list[str]:
    """Public accessor for the OPERATOR_SUGGESTIONS table (test-friendly)."""
    return OPERATOR_SUGGESTIONS.get(bucket, [])


def bucket_gaps(counts: dict[str, int]) -> list[str]:
    """Return bucket names ordered by weakness (smallest count first).

    For ties, the read-only system's `modification` bucket is preferred
    to appear *early* (the coach is told not to lean on DML/DDL).
    """
    def sort_key(b: str) -> tuple[int, int]:
        c = counts.get(b, 0)
        # modification: even when tied with other zeros, surface it first
        # so the coach actively avoids DML
        tiebreak = 0 if b == "modification" else 1
        return (c, tiebreak)

    return sorted(counts.keys(), key=sort_key)




# ---------------------------------------------------------------------------
# Schema graph & traversals (paper §3, A.2)
# ---------------------------------------------------------------------------

@dataclass
class Traversal:
    tables: list[TableRecord]
    label: str

    def serialize(self) -> str:
        """Human-readable schema summary of the traversal path."""
        blocks = []
        for t in self.tables:
            cols = ", ".join(c.name for c in t.columns[:18])
            pk = ", ".join(t.primary_key) or "?"
            fks = [
                f"{fk.columns[0]} -> {fk.ref_table}({fk.ref_columns[0]})"
                for fk in t.foreign_keys
                if fk.columns and fk.ref_columns
            ]
            block = (
                f"TABLE {t.schema}.{t.name} ({t.warehouse_role}, ~{t.row_estimate} rows)\n"
                f"  PK: {pk}\n  Columns: {cols}\n"
                + (f"  FKs: {'; '.join(fks)}\n" if fks else "")
                + (
                    f"  Sample values: "
                    + "; ".join(
                        f"{c.name}: {c.sample_values[:5]}"
                        for c in t.columns
                        if c.sample_values
                    )
                    + "\n"
                    if any(c.sample_values for c in t.columns)
                    else ""
                )
            )
            blocks.append(block)
        return "\n".join(blocks)


def build_schema_graph(tables: list[TableRecord]) -> dict[str, set[str]]:
    """Adjacency: bare table name -> bare names of FK-linked tables."""
    graph: dict[str, set[str]] = {}
    for t in tables:
        bare = t.name
        graph.setdefault(bare, set())
        for fk in t.foreign_keys:
            if fk.ref_table:
                graph[bare].add(fk.ref_table.split(".")[-1])
                graph.setdefault(fk.ref_table.split(".")[-1], set()).add(bare)
        for fk in t.referenced_by:
            if fk.source:
                graph[bare].add(fk.source.split(".")[-1])
    return graph


def random_walk_traversals(
    tables: list[TableRecord],
    n_walks: int,
    max_len: int = 4,
    rng: random.Random | None = None,
) -> list[Traversal]:
    """Fixed-length random walks over the FK graph starting from fact-biased nodes."""
    rng = rng or random.Random()
    by_bare = {t.name: t for t in tables}
    graph = build_schema_graph(tables)

    # bias starts toward fact tables (paper biases toward interesting subgraphs)
    facts = [t for t in tables if getattr(t, "warehouse_role", "") == "fact"]
    starts = [t.name for t in (facts or tables)]

    traversals: list[Traversal] = []
    seen_signatures: set[tuple[str, ...]] = set()

    attempts = 0
    while len(traversals) < n_walks and attempts < n_walks * 10:
        attempts += 1
        start = rng.choice(starts)
        walk = [start]
        visited = {start}
        current = start
        while len(walk) < max_len:
            neighbors = [
                nb
                for nb in graph.get(current, ())
                if nb in by_bare and nb not in visited
            ]
            if not neighbors:
                break
            nxt = rng.choice(neighbors)
            walk.append(nxt)
            visited.add(nxt)
            current = nxt

        sig = tuple(walk)
        if len(walk) < 2 or sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        recs = [by_bare[name] for name in walk if name in by_bare]
        traversals.append(
            Traversal(tables=recs, label=" -> ".join(walk))
        )
    return traversals


# ---------------------------------------------------------------------------
# ICRL synthetic question refinement (paper §3.1)
# ---------------------------------------------------------------------------

BASE_GEN_SYSTEM = """You are generating realistic training questions for a Text-to-SQL assistant.
Given database table schemas (with foreign keys and sample values), write ONE natural
language business question a real analyst would ask, that REQUIRES joining ALL the listed
tables, plus the SQL that answers it.
Rules:
- Output ONLY JSON: {"question": "...", "sql": "..."}
- Read-only SQL (SELECT/WITH). Single statement.
- Use exact table/column names from the schemas.
- The question must be answerable ONLY via these joined tables (not trivially from one).
- Prefer realistic measures, filters, groupings, and ordering."""

FEEDBACK_SYSTEM = """You coach a question generator to produce HARDER, more realistic analyst questions.
You get the schema, the previous generated question+SQL, the per-bucket complexity
counts, and a list of weak buckets with their operator-suggestion tables (paper §A.3).
Your job: write ONE short instruction (max 30 words) that names the *weakest* bucket
and suggests a specific operator upgrade from its suggestions list.
Example: "weakest bucket is aggregation; add a HAVING filter on a SUM group" or
"weakest is conditional; introduce a CASE WHEN ... THEN ... END to bucket rows".
Output ONLY JSON: {"feedback": "..."}"""

REFINE_SYSTEM = BASE_GEN_SYSTEM + """
Incorporate the coaching feedback to produce a MORE COMPLEX question than before."""


@dataclass
class ICRLResult:
    question: str
    sql: str
    tables: list[str]
    reward: float
    iterations: int
    history: list[dict] = field(default_factory=list)


class ICRLGenerator:
    def __init__(
        self,
        llm: LLMClient,
        max_iterations: int = 3,
        plateau_epsilon: float = 0.5,
        sql_dialect: str = "",
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.plateau_epsilon = plateau_epsilon
        self.sql_dialect = sql_dialect

    def _generate(self, traversal: Traversal, extra_context: str = "") -> dict:
        messages = [
            {"role": "system", "content": REFINE_SYSTEM if extra_context else BASE_GEN_SYSTEM},
            {
                "role": "user",
                "content": traversal.serialize() + ("\n\n" + extra_context if extra_context else ""),
            },
        ]
        parsed = self.llm.chat_json(messages, max_tokens=1200)
        return {
            "question": (parsed or {}).get("question", "").strip(),
            "sql": (parsed or {}).get("sql", "").strip(),
        }

    def _feedback(self, traversal: Traversal, question: str, sql: str, counts: dict) -> str:
        gaps = bucket_gaps(counts)
        # present the coach with the gaps + per-bucket operator suggestions
        # for the *weakest* buckets (top 2 by default)
        weak_block_lines = ["BUCKET GAPS (weakest first):"]
        for gap_bucket in gaps:
            count = counts.get(gap_bucket, 0)
            weak_block_lines.append(f"  - {gap_bucket}: count={count}")
        weak_block_lines.append("")
        weak_block_lines.append("OPERATOR SUGGESTIONS for the weakest buckets:")
        for gap_bucket in gaps[:2]:
            sugs = OPERATOR_SUGGESTIONS.get(gap_bucket, [])
            if not sugs:
                continue
            weak_block_lines.append(f"  {gap_bucket}:")
            for s in sugs:
                weak_block_lines.append(f"    - {s}")

        weak_block = "\n".join(weak_block_lines)

        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"SCHEMAS:\n{traversal.serialize()}\n\n"
                    f"PREVIOUS QUESTION:\n{question}\n\n"
                    f"PREVIOUS SQL:\n{sql}\n\n"
                    f"COMPLEXITY COUNTS: {json.dumps(counts)}\n\n"
                    f"{weak_block}\n\n"
                    "Focus your coaching on the weakest bucket(s)."
                ),
            },
        ]
        parsed = self.llm.chat_json(messages, max_tokens=120)
        return (parsed or {}).get("feedback", "")

    def run(
        self,
        traversal: Traversal,
        validate_sql=None,
        min_reward: float = 6.0,
    ) -> ICRLResult | None:
        """Run the ICRL loop for one traversal. Returns best result or None.

        Termination conditions (in order of priority):
          1. SQL fails sqlglot parse → retry with parse-error feedback
          2. `validate_sql` rejects the SQL → retry with sqlguard feedback
          3. reward ≥ `min_reward` → success
          4. plateau detected (Δ-reward < `plateau_epsilon`) → stop
          5. reached `max_iterations` → stop
        """
        context = ""
        best: ICRLResult | None = None
        best_so_far: float = float("-inf")
        history: list[dict] = []

        for it in range(1, self.max_iterations + 1):
            gen = self._generate(traversal, context)
            q, sql = gen["question"], gen["sql"]
            if not q or not sql:
                continue

            # 1) executability gate (sqlglot) — prevents malformed SQL from
            #    scoring artificially high by keyword density.
            try:
                parse_sql(sql, dialect=self.sql_dialect)
            except SQLParseError as exc:
                logger.info("icrl: iteration %d rejected by sqlglot: %s", it, exc)
                history.append({"iteration": it, "error": "parse_error", "detail": str(exc)})
                context = (
                    "FEEDBACK: your SQL did not parse. "
                    "Write a syntactically correct single SELECT statement."
                )
                continue

            # 2) static validation keeps the KB clean (read-only guarantee)
            if validate_sql is not None:
                try:
                    sql = validate_sql(sql)
                except Exception:
                    logger.info("icrl: iteration %d rejected by sqlguard", it)
                    context = (
                        "FEEDBACK: your SQL contained forbidden write/DDL operations. "
                        "Generate strictly read-only SELECT/WITH queries."
                    )
                    continue

            reward, counts = complexity_reward(sql)
            history.append({"iteration": it, "question": q, "sql": sql, "reward": reward})
            logger.info("icrl iter=%d reward=%s tables=%s", it, reward, traversal.label)

            if best is None or reward > best.reward:
                best = ICRLResult(
                    question=q,
                    sql=sql,
                    tables=[t.schema + "." + t.name for t in traversal.tables],
                    reward=reward,
                    iterations=it,
                )

            # 3) success — achieved min reward
            if reward >= min_reward:
                break

            # 4) plateau detection — stop when improvement < ε.
            #    On the first valid iteration, best_so_far is -inf and
            #    we always have a real improvement, so plateau can only
            #    fire from iter 2 onwards. If the reward DECREASES below
            #    what we saw, that's also a plateau (no progress possible).
            is_plateau = it > 1 and (reward - best_so_far) < self.plateau_epsilon
            best_so_far = max(best_so_far, reward)
            if is_plateau:
                logger.info(
                    "icrl: plateau detected iter=%d (Δ=%.3f < ε=%.3f)",
                    it, reward - (best_so_far - reward), self.plateau_epsilon,
                )
                break

            # 5) max_iterations is the natural stop
            if it >= self.max_iterations:
                break

            try:
                fb = self._feedback(traversal, q, sql, counts)
            except LLMError:
                break
            context = f"PREVIOUS ATTEMPT:\n{q}\n{sql}\n\nCOACH FEEDBACK: {fb}"

        return best


# ---------------------------------------------------------------------------
# Knowledge base indexing (schema routing collection)
# ---------------------------------------------------------------------------

ROUTER_SUFFIX = "-router"

# Router KB gating thresholds (paper §3 + plan §1.3).
MIN_ROUTER_REWARD: float = 2.0    # below this, entry is too low-quality to route
MAX_ROUTER_DISTANCE: float = 0.85  # chroma cosine distance; higher = irrelevant


def _deterministic_router_id(question: str, tables: list[str]) -> str:
    """Stable ID for a (question, tables) pair.

    sha1(`<tables_sorted>::<question_normalized>`)[:16] — short enough to be
    readable in logs, long enough to avoid collisions on the ICRL KB scale
    (10k-100k entries).
    """
    sorted_tables = sorted(t.strip().lower() for t in tables)
    q_norm = re.sub(r"\s+", " ", question.strip().lower())
    payload = "::".join(sorted_tables) + "::" + q_norm
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"route:{digest}"


def index_qa_triplets(
    store_any,
    results: Iterable["ICRLResult"],
    *,
    dialect: str = "",
) -> int:
    """Index (question -> tables) pairs into '<collection>-router' for table routing.

    IDs are deterministic (sha1 of `tables_sorted::q_normalized`) so re-runs
    dedupe naturally. Metadata includes the per-bucket reward counts,
    an `executable` flag (last-known parse result), and the `dialect`.
    """
    n = 0
    batch_ids, batch_docs, batch_metas, batch_embeds_src = [], [], [], []
    for r in results:
        # re-compute the reward buckets from the SQL so the metadata is
        # in sync with the latest run (the ICRLResult may not carry the
        # raw counts).
        _, counts = complexity_reward(r.sql)
        try:
            parse_sql(r.sql, dialect=dialect or "")
            executable = True
        except SQLParseError:
            executable = False

        doc = (
            f"Q: {r.question}\nTables: {', '.join(r.tables)}\n"
            f"SQL: {r.sql}\nReward: {r.reward}"
        )
        qid = _deterministic_router_id(r.question, r.tables)
        meta = {
            "kind": "routing",
            "tables": json.dumps(r.tables),
            "reward": float(r.reward),
            "reward_buckets": json.dumps(counts),
            "executable": bool(executable),
            "dialect": dialect or "",
            "traversal_signature": "::".join(sorted(r.tables)),
            "iteration_count": int(r.iterations),
        }
        batch_ids.append(qid)
        batch_docs.append(doc)
        batch_metas.append(meta)
        batch_embeds_src.append(r.question)
        n += 1

    if not batch_ids:
        return 0
    embedder = store_any.embedder
    embeddings = embedder(batch_embeds_src)
    store_any.collection.upsert(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_metas,
        embeddings=embeddings,
    )
    return n


def retrieve_tables_for_question(
    pipeline_store: Any, question: str, top_k: int = 5, chroma_dir: str = "", collection: str = ""
) -> list[str]:
    """Look up similar synthetic questions and return their table sets (router).

    Queries the '<collection>-router' KB; returns [] when it doesn't exist yet.
    """
    import os

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    try:
        client = chromadb.PersistentClient(
            path=chroma_dir or "/tmp",
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        col = client.get_collection((collection or "c") + ROUTER_SUFFIX)
        embedding = pipeline_store.embedder([question])[0]
        res = col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
    except Exception:
        return []

    from collections import Counter

    counter: Counter = Counter()
    # chroma returns lists-of-lists per query
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for meta, d in zip(metas, dists):
        try:
            tables = json.loads((meta or {}).get("tables", "[]"))
        except Exception:
            continue
        weight = max(0.0, 1.0 - float(d))
        for t in tables:
            counter[t] += weight  # weighted vote
    return [t for t, _ in counter.most_common(top_k)]


def retrieve_synthetic_example(
    question: str, embedder, chroma_dir: str, collection: str, max_distance: float = 0.9
) -> dict | None:
    """Nearest synthetic (question, sql, tables) from the router KB, for few-shot.

    Returns {"question", "sql", "tables"} or None.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        col = client.get_collection(collection + ROUTER_SUFFIX)
        if col.count() == 0:
            return None
        embedding = embedder([question])[0]
        res = col.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        dist = (res.get("distances") or [[None]])[0][0]
        if dist is None or float(dist) > max_distance:
            return None
        doc = ((res.get("documents") or [[]])[0][0]) or ""
        meta = ((res.get("metadatas") or [[]])[0][0]) or {}
        sql = ""
        for line in doc.splitlines():
            if line.startswith("SQL:"):
                sql = line[len("SQL:"):].strip()
        return {
            "question": meta.get("question") or doc.splitlines()[0][3:],
            "sql": sql,
            "tables": json.loads(meta.get("tables", "[]")),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reusable orchestration (used by CLI + webapp)
# ---------------------------------------------------------------------------

def get_router_collection(settings, embedder):
    """Open (or create) the '<collection>-router' collection for a profile's store."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=settings.collection + ROUTER_SUFFIX,
        metadata={"embedder": embedder.tag, "kind": "router"},
    )


def run_icrl_generation(settings, llm, n: int = 30, max_iterations: int = 3,
                        validate_sql=None, min_reward: float = 2.0):
    """Full pipeline: introspect → walk → ICRL refine → return results.

    `settings` must carry a resolvable db_dialect (host settings or profile-based).
    """
    from .dialects import get_dialect

    dialect = get_dialect(settings)
    tables_map = dialect.introspect()
    tables = [t for t in tables_map.values() if t.kind == "r"]

    graph = build_schema_graph(tables)
    n_edges = sum(len(v) for v in graph.values())
    logger.info("icrl: graph %d nodes / %d FK edges", len(graph), n_edges)

    traversals = random_walk_traversals(tables, n_walks=n)
    gen = ICRLGenerator(llm, max_iterations=max_iterations)

    results = []
    for i, tr in enumerate(traversals, start=1):
        logger.info("icrl: [%d/%d] %s", i, len(traversals), tr.label)
        try:
            res = gen.run(tr, validate_sql=validate_sql, min_reward=min_reward)
        except Exception as exc:
            logger.warning("icrl: traversal %s skipped (%s)", tr.label, exc)
            continue
        if res is not None and res.reward >= 2.0:
            results.append(res)
    return results


def index_results_for_profile(settings, embedder, results) -> int:
    """Index ICRL results into the profile's router collection."""
    from .store import VectorStore  # noqa: F401  (embedder tag contract)

    class _RouterStore:
        pass

    rs = _RouterStore()
    rs.embedder = embedder
    rs.collection = get_router_collection(settings, embedder)
    return index_qa_triplets(rs, results)
