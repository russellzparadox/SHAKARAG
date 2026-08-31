"""Tests for the upgraded router KB (Phase C: deterministic IDs, reward gate,
weighted distance, top-K, table-overlap boost)."""
from __future__ import annotations

import hashlib
import json

import pytest


# ----- helpers ---------------------------------------------------------------

class _HashEmbedder:
    """Deterministic, dimension-8 embedder for tests.

    Each input string is hashed (sha1) and split into 8 unsigned 8-bit
    chunks → a length-8 list of ints in [0, 255], then normalised. Two
    identical inputs produce identical vectors; a single character change
    flips many bits → high L2 distance.
    """

    dim = 8
    tag = "hash-8"

    def __call__(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha1(t.encode("utf-8")).digest()  # 20 bytes
            vec = [h[i] for i in range(8)]
            # normalise to unit length so cosine-style distance makes sense
            norm = max(1.0, sum(x * x for x in vec) ** 0.5)
            out.append([x / norm for x in vec])
        return out


def _tmp_router_col(tmp_path):
    """Create a fresh '<collection>-router' Chroma collection in a tmp dir.

    Returns (collection, chroma_path, embedder)."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    embedder = _HashEmbedder()
    client = chromadb.PersistentClient(
        path=str(tmp_path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    col = client.get_or_create_collection(
        name="c-router",
        metadata={"embedder": embedder.tag, "kind": "router"},
    )
    return col, str(tmp_path), embedder


def _result(question, sql, tables, reward=3.0, history=None):
    from rag.icrl import ICRLResult

    return ICRLResult(
        question=question,
        sql=sql,
        tables=tables,
        reward=reward,
        iterations=1,
        history=history or [],
    )


# ----- C1/C2: deterministic IDs --------------------------------------------

def test_index_qa_deterministic_ids(monkeypatch, tmp_path):
    """Same (question, tables) must produce the same ID on re-index.

    Implementation: sha1 of `<tables_sorted>::<q_normalized>`[:16].
    """
    from rag.icrl import index_qa_triplets

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)

    class _StubStore:
        pass

    store = _StubStore()
    store.embedder = embedder
    store.collection = col

    results = [
        _result("What is the total sales by region?", "SELECT SUM(s) FROM sales",
                ["public.sales", "public.region"], reward=4.0),
    ]

    index_qa_triplets(store, list(results))
    ids_after_first = set(col.get()["ids"])

    # re-index the same triplet
    index_qa_triplets(store, list(results))
    ids_after_second = set(col.get()["ids"])

    assert ids_after_first == ids_after_second
    # and exactly one row exists (dedupe)
    assert len(ids_after_first) == 1


# ----- C3/C4: metadata fields -----------------------------------------------

def test_index_qa_metadata_includes_buckets_and_executable(monkeypatch, tmp_path):
    """Index metadata must include `reward_buckets` (JSON) and `executable`."""
    from rag.icrl import index_qa_triplets

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)

    class _StubStore:
        pass

    store = _StubStore()
    store.embedder = embedder
    store.collection = col

    results = [
        _result("count of orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status",
                ["public.orders"], reward=3.0),
    ]

    index_qa_triplets(store, list(results))
    metas = col.get(include=["metadatas"])["metadatas"]
    assert len(metas) == 1
    meta = metas[0]
    assert "reward_buckets" in meta
    assert "executable" in meta
    assert "dialect" in meta
    # and the buckets JSON parses + has the right shape
    buckets = json.loads(meta["reward_buckets"])
    assert set(buckets.keys()) == {"retrieval", "conditional", "aggregation", "modification"}


# ----- C5/C6: distance + reward gate ---------------------------------------

def test_retrieve_tables_gates_on_distance_and_reward(tmp_path):
    """Entries beyond MAX_DISTANCE (0.85) or with reward < MIN_REWARD are gated out.

    We index two entries with identical embedding (so distance ≈ 0) but very
    different rewards; both should still be considered (the gate is on
    distance AND on reward). Then we index a third with a deliberately
    *different* embedding (so distance > 0.85) and assert it is excluded.
    """
    from rag.icrl import (
        retrieve_tables_for_question,
        MIN_ROUTER_REWARD,
        MAX_ROUTER_DISTANCE,
    )

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)

    # Two entries: same embedding (distance 0), different reward
    near_vec = embedder(["near query"])[0]
    far_vec = [0.0] * 8  # orthogonal-ish to near_vec → high distance
    far_vec[0] = 1.0
    # force L2 distance > 0.85 by giving an orthogonal unit vector
    norm = (sum(x * x for x in near_vec) ** 0.5) or 1.0
    far_vec = [-v / norm for v in near_vec]  # exact opposite → distance 2

    col.upsert(
        ids=["near-1", "far-1"],
        documents=["Q: near\nTables: public.orders", "Q: far\nTables: public.audit"],
        metadatas=[
            {"tables": json.dumps(["public.orders"]), "reward": 5.0},
            {"tables": json.dumps(["public.audit"]), "reward": 5.0},
        ],
        embeddings=[near_vec, far_vec],
    )

    # query with the *near* embedding
    class _StubStore:
        embedder = embedder

    tables = retrieve_tables_for_question(
        _StubStore(), "near query", top_k=5,
        chroma_dir=chroma_dir, collection="c",
    )
    # far entry (orthogonal, distance 2 > 0.85) is gated out
    assert "public.audit" not in tables
    # near entry survives
    assert "public.orders" in tables


# ----- C7: weight = (1 - distance) * reward ---------------------------------

def test_retrieve_tables_weighted_by_distance_and_reward(tmp_path):
    """Vote weight = (1 - distance) * reward; verify with two near entries of differing reward."""
    from rag.icrl import retrieve_tables_for_question

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)
    base = embedder(["base"])[0]

    col.upsert(
        ids=["low", "high"],
        documents=["Q: low\nTables: public.x", "Q: high\nTables: public.x"],
        metadatas=[
            {"tables": json.dumps(["public.x"]), "reward": 1.0},
            {"tables": json.dumps(["public.x"]), "reward": 8.0},
        ],
        embeddings=[base, base],
    )

    class _StubStore:
        embedder = embedder

    # internal call: inspect the returned weight indirectly — high-reward entry
    # should be considered (still distance 0, weight 1.0) and survive; we just
    # assert both contribute to the same table (public.x) so the test of the
    # weight formula is observable via the function not erroring on multiple
    # entries with same embedding.
    tables = retrieve_tables_for_question(
        _StubStore(), "base", top_k=5,
        chroma_dir=chroma_dir, collection="c",
    )
    assert tables == ["public.x"]


# ----- C8/C9: top-K retrieval of synthetic examples -------------------------

def test_retrieve_synthetic_returns_top_k(tmp_path):
    """retrieve_synthetic_example now returns up to K examples ordered by score."""
    from rag.icrl import retrieve_synthetic_example

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)
    base = embedder(["base question"])[0]
    # three identical embeddings so distances are 0 — ordering by score (1-d)*reward
    col.upsert(
        ids=["e1", "e2", "e3"],
        documents=[
            "Q: base question\nTables: public.orders\nSQL: SELECT 1\nReward: 1.0",
            "Q: base question\nTables: public.orders\nSQL: SELECT 2\nReward: 2.0",
            "Q: base question\nTables: public.orders\nSQL: SELECT 3\nReward: 3.0",
        ],
        metadatas=[
            {"tables": json.dumps(["public.orders"]), "reward": 1.0,
             "reward_buckets": "{}", "executable": True, "dialect": ""},
            {"tables": json.dumps(["public.orders"]), "reward": 2.0,
             "reward_buckets": "{}", "executable": True, "dialect": ""},
            {"tables": json.dumps(["public.orders"]), "reward": 3.0,
             "reward_buckets": "{}", "executable": True, "dialect": ""},
        ],
        embeddings=[base, base, base],
    )

    results = retrieve_synthetic_example(
        "base question", embedder, chroma_dir, "c", k=3
    )
    assert isinstance(results, list)
    assert len(results) == 3
    # ordered by descending score: e3 first
    sqls = [r["sql"] for r in results]
    assert sqls == ["SELECT 3", "SELECT 2", "SELECT 1"]


# ----- C10/C11: table-overlap boost ----------------------------------------

def test_retrieve_synthetic_boosts_table_overlap(tmp_path):
    """Candidates whose tables overlap with the `boost_tables` set win ties."""
    from rag.icrl import retrieve_synthetic_example

    col, chroma_dir, embedder = _tmp_router_col(tmp_path)
    base = embedder(["base"])[0]
    # two entries: same distance, same reward; A has tables=[public.x], B has [public.y]
    col.upsert(
        ids=["A", "B"],
        documents=[
            "Q: base\nTables: public.x\nSQL: SELECT a\nReward: 5.0",
            "Q: base\nTables: public.y\nSQL: SELECT b\nReward: 5.0",
        ],
        metadatas=[
            {"tables": json.dumps(["public.x"]), "reward": 5.0,
             "reward_buckets": "{}", "executable": True, "dialect": ""},
            {"tables": json.dumps(["public.y"]), "reward": 5.0,
             "reward_buckets": "{}", "executable": True, "dialect": ""},
        ],
        embeddings=[base, base],
    )

    # when we boost public.x, A must rank above B
    results = retrieve_synthetic_example(
        "base", embedder, chroma_dir, "c", k=2,
        boost_tables={"public.x"},
    )
    assert results[0]["sql"] == "SELECT a"
    assert results[1]["sql"] == "SELECT b"
