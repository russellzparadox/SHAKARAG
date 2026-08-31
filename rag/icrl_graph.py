"""SchemaGraph + traversal enumeration — paper Algorithms A.1 and A.2.

Replaces the bare-name graph + random-walk heuristic in `rag/icrl.py` with the
typed, deterministic representation from Toteja et al. (§A.1, §A.2).

Public surface (consumed by `rag/icrl.py` and the test suite):
    - `EdgeType`            : int enum {1=root→db, 2=db→table, 3=table↔table (FK)}
    - `GraphEdge`           : (src, dst, type, is_reverse) named tuple
    - `SchemaGraph`         : node/edge container with adjacency helpers
    - `build_from_tables`   : paper Alg. A.1 — build graph from introspected tables
    - `enumerate_traversals`: paper Alg. A.2 — depth-first enumeration w/ cutoff k
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class EdgeType(IntEnum):
    """Paper Alg. A.1 edge types: 1=root→db, 2=db→table, 3=table→table (FK)."""

    ROOT_TO_DB = 1
    DB_TO_TABLE = 2
    TABLE_FK = 3


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    type: EdgeType
    is_reverse: bool = False  # true for FK back-edges (A←B for an A→B FK)


@dataclass
class SchemaGraph:
    """Container for the typed schema graph (paper Alg. A.1)."""

    nodes: set[str] = field(default_factory=set)
    edges: list[GraphEdge] = field(default_factory=list)
    # adjacency: node -> list of (dst, edge_type) using edges AS-STORED
    _adj: dict[str, list[GraphEdge]] = field(default_factory=lambda: defaultdict(list))

    # ---- mutation ----

    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(self, src: str, dst: str, type: EdgeType, *, is_reverse: bool = False) -> GraphEdge:
        self.add_node(src)
        self.add_node(dst)
        e = GraphEdge(src=src, dst=dst, type=type, is_reverse=is_reverse)
        self.edges.append(e)
        self._adj[src].append(e)
        return e

    # ---- query ----

    def has_node(self, name: str) -> bool:
        return name in self.nodes

    def edges_of_type(self, t: EdgeType) -> list[GraphEdge]:
        return [e for e in self.edges if e.type == t]

    def outgoing(self, node: str) -> list[GraphEdge]:
        """All edges leaving `node` (typed edges only — not their reverses)."""
        return list(self._adj.get(node, ()))


# ---- Algorithm A.1: build graph from introspected TableRecords ----

def _bare(ref: str) -> str:
    """Strip schema prefix from a referenced table name."""
    return ref.split(".")[-1] if ref else ref


def build_from_tables(tables: Iterable) -> SchemaGraph:
    """Build the typed schema graph per paper Algorithm A.1.

    Nodes: one root `R`, one node per database, one node per table.
    Edges:
        1 (R  → DB)        for every distinct database
        2 (DB → Table)     for every table in that database
        3 (Table → Table)  for every FK (directional, ref as declared);
                           a *reverse* navigation edge is also recorded
                           with `is_reverse=True` so the traversal can
                           still walk back along the FK.
    """
    g = SchemaGraph()
    g.add_node("R")

    # group tables by database
    db_to_tables: dict[str, list] = defaultdict(list)
    for t in tables:
        g.add_node(t.name)
        db_to_tables[t.schema].append(t)

    # type-1 + type-2
    for db, ts in db_to_tables.items():
        g.add_node(db)
        g.add_edge("R", db, EdgeType.ROOT_TO_DB)
        for t in ts:
            g.add_edge(db, t.name, EdgeType.DB_TO_TABLE)

    # type-3 (directional, with reverse navigation edge)
    for t in tables:
        for fk in t.foreign_keys:
            dst = _bare(fk.ref_table)
            if not dst or not g.has_node(dst):
                continue
            g.add_edge(t.name, dst, EdgeType.TABLE_FK, is_reverse=False)
            g.add_edge(dst, t.name, EdgeType.TABLE_FK, is_reverse=True)

    return g


# ---- Algorithm A.2: enumerate traversals (paper §A.2) ----
# Stub — full implementation lands in task A7. Returning [] keeps the
# module importable so the A2/A3 graph tests can run today.

def enumerate_traversals(graph: SchemaGraph, cutoff: int = 3, *, min_tables: int = 2):  # noqa: D401
    """Paper Alg. A.2 — depth-first traversal enumeration with cutoff.

    Stub. Returns [] until A7. Cutoff `k` caps path length. `min_tables`
    filters out traversals shorter than 2 tables (paper requires joins).
    """
    return []

