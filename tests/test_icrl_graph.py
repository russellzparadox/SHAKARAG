"""Tests for the faithful SchemaGraph + enumerate_traversals (paper §A.1, §A.2).

Implements Algorithms A.1 (graph construction) and A.2 (serialisation with cutoff)
verbatim from Toteja et al. — replaces the bare-name graph + random-walk heuristic
in `rag/icrl.py` with typed edges and deterministic depth-first enumeration.
"""
from __future__ import annotations

import pytest

# Graph implementation will live in rag.icrl_graph and be re-exported from rag.icrl
from rag.icrl_graph import SchemaGraph, EdgeType, enumerate_traversals, build_from_tables  # noqa: F401
from rag.icrl import build_schema_graph  # legacy (back-compat)  # noqa: F401


# ---- fixtures ----

def _t(name, schema="dbo", role="dimension", fks=None, referenced_by=None):
    """Lightweight TableRecord stand-in — only fields the graph builder needs."""
    from rag.introspect import Column, ForeignKey, TableRecord

    cols = [
        Column(name="id", type="int", nullable=False, default=None,
               comment=None, identity=False, generated=False, pk=True),
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
    for fk in referenced_by or []:
        rec.referenced_by.append(fk)
    return rec


# ---- A2: typed edges (R→DB, DB→Table, Table→Table FK) ----

def test_algorithm_a1_root_db_table_fk_edges():
    """Alg. A.1: graph has typed edges: 1=root→db, 2=db→table, 3=table→table (FK)."""
    # 2 databases (db1, db2), 4 tables (A,B in db1; C,D in db2), 1 FK A→B
    a = _t("A", schema="db1", role="dimension", fks=["B"])
    b = _t("B", schema="db1", role="fact")
    c = _t("C", schema="db2", role="dimension")
    d = _t("D", schema="db2", role="dimension")

    g = build_from_tables([a, b, c, d])

    # nodes: 1 root, 2 databases, 4 tables
    assert g.has_node("R")
    assert g.has_node("db1") and g.has_node("db2")
    for n in ("A", "B", "C", "D"):
        assert g.has_node(n), f"missing node {n}"

    # edges by type
    type1 = g.edges_of_type(EdgeType.ROOT_TO_DB)  # R → DB
    type2 = g.edges_of_type(EdgeType.DB_TO_TABLE)  # DB → Table
    type3 = g.edges_of_type(EdgeType.TABLE_FK)  # Table → Table (FK)
    assert {e.dst for e in type1} == {"db1", "db2"}
    assert {e.dst for e in type2 if e.src == "db1"} == {"A", "B"}
    assert {e.dst for e in type2 if e.src == "db2"} == {"C", "D"}
    # exactly one FK edge in the FK direction (A→B)
    assert {e.src for e in type3 if not e.is_reverse} == {"A"}
    assert {e.dst for e in type3 if not e.is_reverse} == {"B"}


# ---- A4: FK directional + reverse navigation edge ----

def test_algorithm_a1_fk_directional_with_reverse():
    """FK edges are stored directionally (A→B) AND carry a reverse nav edge (B→A, is_reverse=True)."""
    a = _t("A", schema="db1", fks=["B"])
    b = _t("B", schema="db1")
    g = build_from_tables([a, b])

    fk_edges = g.edges_of_type(EdgeType.TABLE_FK)
    # directional: A → B
    directional = [e for e in fk_edges if not e.is_reverse]
    reverse = [e for e in fk_edges if e.is_reverse]
    assert len(directional) == 1 and directional[0].src == "A" and directional[0].dst == "B"
    assert len(reverse) == 1 and reverse[0].src == "B" and reverse[0].dst == "A"
    # reverse is *navigation-only* (not a duplicate FK declaration)
    assert directional[0].type == EdgeType.TABLE_FK == reverse[0].type


# ---- A6: enumerate_traversals with cutoff depth ----

def test_algorithm_a2_traversals_cutoff_depth():
    """Paper Alg. A.2: cutoff k caps traversal length. Path = node list from R.

    Single-DB chain A→B→C→D via type-3. cutoff=2 and min_tables=2
    (default for SQL generation; joins require ≥ 2 tables) yields only
    2-table traversals: (R, db1, A, B), (R, db1, B, C), (R, db1, C, D).
    """
    # chain A→B→C→D
    a = _t("A", schema="db1", fks=["B"])
    b = _t("B", schema="db1", fks=["C"])
    c = _t("C", schema="db1", fks=["D"])
    d = _t("D", schema="db1")
    g = build_from_tables([a, b, c, d])

    traversals = enumerate_traversals(g, cutoff=2, min_tables=2)
    paths = {tuple(t.nodes) for t in traversals}

    expected = {
        ("R", "db1", "A", "B"),
        ("R", "db1", "B", "C"),
        ("R", "db1", "C", "D"),
    }
    assert paths == expected


# ---- A8: type-3 power-set branching at fan-out ----

def test_algorithm_a2_power_set_branching():
    """Paper Alg. A.2 lines 10-15: a type-3 fan-out with >1 children expands
    via the power set, yielding non-trivial multi-child traversals.

    Hub T with 3 dim children D1, D2, D3; cutoff=2:
      expected 2-table traversals cover each child individually,
      and 3-table traversals cover (D1,D2), (D1,D3), (D2,D3) — the
      non-empty, non-full power-set subsets.
    """
    hub = _t("Hub", schema="db1", role="fact", fks=["D1", "D2", "D3"])
    d1 = _t("D1", schema="db1")
    d2 = _t("D2", schema="db1")
    d3 = _t("D3", schema="db1")
    g = build_from_tables([hub, d1, d2, d3])

    traversals = enumerate_traversals(g, cutoff=2, min_tables=2)
    paths = {tuple(t.nodes) for t in traversals}

    # 2-table: (R, db1, Hub, D*) for each child
    two_table = {("R", "db1", "Hub", c) for c in ("D1", "D2", "D3")}
    # 3-table: (R, db1, Hub, X, Y) for each (X,Y) pair from {D1,D2,D3}
    three_table = {("R", "db1", "Hub", x, y) for x, y in [("D1", "D2"), ("D1", "D3"), ("D2", "D3")]}
    assert two_table.issubset(paths), f"missing 2-table: {two_table - paths}"
    assert three_table.issubset(paths), f"missing 3-table: {three_table - paths}"
