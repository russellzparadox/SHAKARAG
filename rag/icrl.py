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

logger = logging.getLogger("chat.icrl")

# ---------------------------------------------------------------------------
# Complexity reward (paper §A.3)
# ---------------------------------------------------------------------------

BUCKETS: dict[str, dict[str, int]] = {
    # B1 data retrieval & filtering
    "retrieval": {
        "SELECT": 1, "FROM": 1, "JOIN": 2, "INNER JOIN": 3, "LEFT JOIN": 3,
        "RIGHT JOIN": 3, "FULL JOIN": 3, "ON": 2, "WHERE": 2, "GROUP BY": 3,
        "HAVING": 3, "ORDER BY": 2, "DISTINCT": 2, "LIMIT": 1, "TOP": 1,
    },
    # B2 data modification (discouraged — we only generate read-only)
    "modification": {
        "INSERT": 2, "UPDATE": 3, "DELETE": 4, "DROP": 5, "ALTER": 5,
    },
    # B3 conditional logic
    "conditional": {
        "AND": 1, "OR": 1, "NOT": 1, "IN (": 2, "BETWEEN": 2, "LIKE": 2,
        "CASE WHEN": 3, "WHEN": 2, "THEN": 2, "ELSE": 2, "EXISTS": 3,
    },
    # B4 aggregation & sorting
    "aggregation": {
        "AVG(": 3, "SUM(": 3, "COUNT(": 3, "MIN(": 3, "MAX(": 3,
        "ASC": 1, "DESC": 1, "COALESCE": 2, "DATEPART": 2, "YEAR(": 2, "MONTH(": 2,
    },
}

NEGATIVE_REWARD = {"modification"}  # read-only system: DML keywords reduce reward


def complexity_reward(sql: str) -> tuple[float, dict[str, int]]:
    """Paper eq. (1)+(2): weighted bucket-frequency reward. Returns (reward, details)."""
    upper = f" {sql.upper()} "
    bucket_scores: dict[str, float] = {}
    counts: dict[str, int] = {}

    total_weighted = 0.0
    for bucket, keywords in BUCKETS.items():
        freq = 0
        for kw, weight in keywords.items():
            n = upper.count(kw)
            if n:
                freq += n * weight
        counts[bucket] = freq
        bucket_scores[bucket] = freq
        total_weighted += freq

    if total_weighted == 0:
        return 0.0, {k: 0 for k in counts}

    # normalise each bucket share then apply bucket weights:
    # retrieval & aggregation are desirable; conditional adds complexity;
    # modification is penalised (read-only system).
    weights = {"retrieval": 1.0, "conditional": 1.2, "aggregation": 1.5, "modification": -2.0}
    reward = sum(
        weights.get(b, 1.0) * (freq / total_weighted) * total_weighted / 10.0
        for b, freq in bucket_scores.items()
    )
    return round(reward, 3), counts


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
You get the schema, the previous generated question+SQL, and its complexity reward breakdown.
Write ONE short instruction (max 30 words) describing how to make the next question more
complex and realistic, e.g.: "add a HAVING filter on the aggregate", "compare two periods",
"add CASE-based categorisation", "require a LEFT JOIN to include entities with zero rows".
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
    def __init__(self, llm: LLMClient, max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations

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
        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"SCHEMAS:\n{traversal.serialize()}\n\n"
                    f"PREVIOUS QUESTION:\n{question}\n\n"
                    f"PREVIOUS SQL:\n{sql}\n\n"
                    f"COMPLEXITY COUNTS: {json.dumps(counts)}\n"
                    "Low aggregation/conditional counts mean too simple."
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
        """Run the ICRL loop for one traversal. Returns best result or None."""
        context = ""
        best: ICRLResult | None = None
        history: list[dict] = []

        for it in range(1, self.max_iterations + 1):
            gen = self._generate(traversal, context)
            q, sql = gen["question"], gen["sql"]
            if not q or not sql:
                continue

            # static validation keeps the KB clean (read-only guarantee)
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

            if reward >= min_reward or it >= self.max_iterations:
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


def index_qa_triplets(store_any, results: Iterable[ICRLResult]) -> int:
    """Index (question -> tables) pairs into '<collection>-router' for table routing."""
    n = 0
    batch_ids, batch_docs, batch_metas, batch_embeds_src = [], [], [], []
    for r in results:
        doc = (
            f"Q: {r.question}\nTables: {', '.join(r.tables)}\n"
            f"SQL: {r.sql}\nReward: {r.reward}"
        )
        qid = "route:" + re.sub(r"[^a-z0-9]+", "-", r.question.lower())[:80]
        meta = {"kind": "routing", "tables": json.dumps(r.tables), "reward": r.reward}
        batch_ids.append(qid)
        batch_docs.append(doc)
        batch_metas.append(meta)
        batch_embeds_src.append(r.question)
        n += 1

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
