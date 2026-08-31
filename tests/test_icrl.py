"""Tests for the ICRL synthetic-QA module (paper method)."""
import json

import pytest

from rag.icrl import (
    ICRLGenerator,
    Traversal,
    build_schema_graph,
    complexity_reward,
    random_walk_traversals,
)
from rag.pipeline import RagPipeline


# ---------- reward ----------

def test_reward_simple_select_low():
    r, _ = complexity_reward("SELECT a FROM t")
    assert r < 1.0


def test_reward_complex_join_aggregate_high():
    sql = (
        "SELECT d.Title, SUM(f.Amount) AS total FROM f "
        "JOIN d ON f.sid = d.sid WHERE f.year = 2024 "
        "GROUP BY d.Title HAVING SUM(f.Amount) > 100 ORDER BY total DESC"
    )
    r, counts = complexity_reward(sql)
    assert r > 2.0
    assert r > complexity_reward("SELECT a FROM t")[0] * 5
    assert counts["aggregation"] >= 6
    assert counts["retrieval"] >= 10


def test_reward_penalises_dml():
    _, counts = complexity_reward("DELETE FROM t")
    assert counts["modification"] > 0


def test_empty_sql_zero():
    assert complexity_reward("")[0] == 0.0


# ---------- graph & traversals ----------

def _table(name, schema="dbo", role="dimension", fks=None):
    from rag.introspect import Column, ForeignKey, TableRecord

    cols = [
        Column(name="id", type="int", nullable=False, default=None,
               comment=None, identity=False, generated=False, pk=True),
        Column(name="title", type="nvarchar", nullable=True, default=None,
               comment=None, identity=False, generated=False),
        Column(name="fk_id", type="int", nullable=True, default=None,
               comment=None, identity=False, generated=False),
    ]
    rec = TableRecord(
        schema=schema,
        name=name,
        kind="r",
        row_estimate=1000,
        comment=None,
        columns=cols,
        primary_key=["id"],
        warehouse_role=role,
    )
    for fk_target in fks or []:
        rec.foreign_keys.append(
            ForeignKey(name=f"fk_{name}_{fk_target}", columns=["fk_id"],
                       ref_table=fk_target, ref_columns=["id"])
        )
    return rec


def test_schema_graph_fk_edges():
    t1 = _table("Fact", role="fact", fks=["Dim"])
    t2 = _table("Dim")
    graph = build_schema_graph([t1, t2])
    assert "Dim" in graph["Fact"]
    assert "Fact" in graph["Dim"]  # undirected adjacency


def test_random_walks_unique_and_connected():
    tables = [
        _table("FactSales", role="fact", fks=["DimDate", "DimCustomer"]),
        _table("DimDate"),
        _table("DimCustomer", fks=["DimRegion"]),
        _table("DimRegion"),
        _table("Orphan"),
    ]
    trs = random_walk_traversals(tables, n_walks=6, max_len=4, rng=__import__("random").Random(42))
    assert len(trs) >= 2
    sigs = {tuple(t.name for t in tr.tables) for tr in trs}
    assert len(sigs) == len(trs)  # unique walks
    for tr in trs:
        assert len(tr.tables) >= 2


def test_traversal_serialize_contains_schema_info():
    t1 = _table("Fact", role="fact", fks=["Dim"])
    t2 = _table("Dim")
    tr = Traversal(tables=[t1, t2], label="Fact -> Dim")
    s = tr.serialize()
    assert "dbo.Fact" in s and "PK:" in s and "FKs:" in s


# ---------- ICRL loop ----------

class FakeLLM:
    """Returns queued JSON responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)

    def chat_json(self, messages, max_tokens=8192):
        return json.loads(self.responses.pop(0))

    def chat(self, messages, max_tokens=8192):
        return "ok"


def _traversal():
    return Traversal(tables=[_table("Fact", role="fact", fks=["Dim"]), _table("Dim")],
                     label="Fact -> Dim")


def test_icrl_stops_at_min_reward(tmp_path):
    good = json.dumps({
        "question": "total per supplier",
        "sql": ("SELECT d.title, SUM(f.amount) AS t FROM fact f JOIN dim d "
                "ON f.fk_id = d.id GROUP BY d.title HAVING SUM(f.amount) > 5 "
                "ORDER BY t DESC"),
    })
    llm = FakeLLM([good])
    gen = ICRLGenerator(llm, max_iterations=3)
    res = gen.run(_traversal(), validate_sql=lambda s: s, min_reward=2.0)
    assert res is not None
    assert res.iterations == 1  # reward met immediately
    assert len(llm.responses) == 0


def test_icrl_refines_on_feedback(tmp_path):
    simple = json.dumps({"question": "list rows", "sql": "SELECT a FROM Fact"})
    feedback = json.dumps({"feedback": "add GROUP BY and SUM aggregation"})
    better = json.dumps({
        "question": "sum amount per category",
        "sql": ("SELECT cat, SUM(a) FROM fact JOIN dim ON x GROUP BY cat "
                "ORDER BY SUM(a) DESC"),
    })
    llm = FakeLLM([simple, feedback, better])
    gen = ICRLGenerator(llm, max_iterations=2)
    res = gen.run(_traversal(), validate_sql=lambda s: s, min_reward=50.0)
    assert res is not None
    assert res.iterations == 2
    assert res.reward > complexity_reward("SELECT a FROM Fact")[0]


def test_icrl_rejects_dml(tmp_path):
    calls = {"n": 0}

    class GuardLLM:
        def chat_json(self, messages, max_tokens=8192):
            calls["n"] += 1
            return {"question": "delete stuff", "sql": "DELETE FROM Fact"}

        def chat(self, messages, max_tokens=8192):
            return ""

    gen = ICRLGenerator(GuardLLM(), max_iterations=2)
    res = gen.run(_traversal(),
                  validate_sql=lambda s: (_ for _ in ()).throw(ValueError("DML")),
                  min_reward=1.0)
    # both iterations rejected by guard → no result
    assert res is None
    assert calls["n"] == 2


# ---------- pipeline integration ----------

class FakeStore:
    embedder = None

    def query_text(self, text, top_k):
        return []

    def all_table_metas(self):
        return []


def test_join_map_extracts_conditions():
    p = RagPipeline.__new__(RagPipeline)
    context = (
        "JOIN MAP ignored\n"
        "BI.FactLogisticControl (table) columns part 1/4:\n"
        "- SupplierID int [FK->DimSupplier(SupplierID)]\n"
        "- CompanyID int [FK->DimCompany(CompanyID)]\n"
    )
    jm = p._join_map(["FactLogisticControl"], context)
    # single table → no map
    assert jm == ""


def test_join_map_two_tables():
    p = RagPipeline.__new__(RagPipeline)
    context = (
        "BI.FactLogisticControl (table) columns part 1/4:\n"
        "- SupplierID int [FK->DimSupplier(SupplierID)]\n"
        "\n---\n\n"
        "BI.DimSupplier (table) columns:\n"
        "- SupplierID int\n"
        "- Title nvarchar\n"
    )
    jm = p._join_map(["FactLogisticControl", "DimSupplier"], context)
    assert "FactLogisticControl.SupplierID = DimSupplier.SupplierID" in jm
