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

from itertools import chain, combinations


@dataclass(frozen=True)
class TraversalPath:
    """A path through the schema graph (R, db, table, table, ...).

    `nodes` is the ordered node list; `edges` is the parallel list of edge
    types that were traversed to reach each node (length = len(nodes) - 1).
    `tables` is a convenience accessor (nodes minus the structural prefix).
    """

    nodes: tuple[str, ...]

    @property
    def tables(self) -> tuple[str, ...]:
        # first two nodes are R and the database; everything after is a table
        return self.nodes[2:]


def _power_set(items: list) -> list[tuple]:
    """All non-empty, non-full subsets of `items`, preserving order.

    Paper Alg. A.2 lines 11–15: at a type-3 fan-out with >1 children, expand
    via the power set; the "empty" and "full" subsets are skipped because
    the DFS already recurses into the single-child case implicitly.
    """
    if len(items) < 2:
        return [tuple(items)] if items else []
    return [
        s
        for r in range(1, len(items) + 1)
        for s in combinations(items, r)
        if 0 < len(s) < len(items)
    ]


def enumerate_traversals(
    graph: SchemaGraph, cutoff: int = 3, *, min_tables: int = 2
) -> list[TraversalPath]:
    """Paper Alg. A.2 — depth-first traversal enumeration with cutoff.

    Starts from root `R`, follows type-1 (root→db), type-2 (db→table), and
    type-3 (table↔table FK) edges, producing all paths of length ≤ `cutoff`
    *table* nodes. Per-path `visited` set prevents cycles.

    At each type-3 fan-out (>1 children), the algorithm expands via the
    power set of children (paper lines 10–15), so a hub table with 3 dims
    yields both (1,2) and (2,3) subsets as parallel exploration.

    `min_tables` filters out traversals shorter than N tables — paper Eq. 1+
    requires joins, so single-table paths are not useful for SQL generation.
    """
    out: list[TraversalPath] = []
    seen: set[tuple[str, ...]] = set()
    visited: set[str] = set()

    def _walk(node: str, path: tuple[str, ...], table_depth: int) -> None:
        # table_depth = number of *table* nodes in the path after this visit
        if node in visited or table_depth > cutoff:
            return
        visited.add(node)
        path = path + (node,)

        # count tables in current path (everything after the first 2 nodes
        # R, db is a table)
        tables_in_path = path[2:]
        if len(tables_in_path) >= min_tables and path not in seen:
            seen.add(path)
            out.append(TraversalPath(nodes=path))

        # collect "forward" children:
        #   - type-1: R → DB (no table increment)
        #   - type-2: DB → TABLE (+1)
        #   - type-3 (directional): TABLE → TABLE (+1)
        #   - reverse FK edges are NOT walked (visited set already blocks
        #     cycles; reverse edges would just generate mirror paths)
        # At type-3 fan-outs with >1 forward children, expand via the
        # paper's power set (Alg. A.2 lines 10–15).
        for e in graph.outgoing(node):
            if e.is_reverse:
                continue
            if e.type in (EdgeType.DB_TO_TABLE, EdgeType.TABLE_FK):
                step_depth = table_depth + 1
            else:
                step_depth = table_depth

            if e.type == EdgeType.TABLE_FK:
                # collect forward FK siblings at this node
                siblings = [
                    o.dst
                    for o in graph.outgoing(node)
                    if o.type == EdgeType.TABLE_FK and not o.is_reverse and o.dst not in visited
                ]
                if len(siblings) > 1:
                    # paper Alg. A.2 power-set expansion
                    for subset in _power_set(siblings):
                        # recurse into the first child (counts as +1 table)
                        new_path = path + (subset[0],)
                        if new_path[2:].count(subset[0]) <= 1:
                            yield_branch(new_path, table_depth + 1)
                        # and recurse into the rest as children of the first
                        for s in subset[1:]:
                            if s in visited:
                                continue
                            visited.add(s)
                            sub_path = new_path + (s,)
                            if (
                                len(sub_path[2:]) >= min_tables
                                and sub_path not in seen
                            ):
                                seen.add(sub_path)
                                out.append(TraversalPath(nodes=sub_path))
                            visited.discard(s)
                    continue
            _walk(e.dst, path, step_depth)

        visited.discard(node)

    def yield_branch(p: tuple[str, ...], td: int) -> None:
        """Emit `p` if eligible, then recurse into the last node's forward children."""
        if td > cutoff:
            return
        if len(p[2:]) >= min_tables and p not in seen:
            seen.add(p)
            out.append(TraversalPath(nodes=p))
        # recurse from the last node in p, mirroring _walk's body
        last = p[-1]
        for e in graph.outgoing(last):
            if e.is_reverse or e.dst in visited:
                continue
            step = td + 1 if e.type in (EdgeType.DB_TO_TABLE, EdgeType.TABLE_FK) else td
            _walk(e.dst, p, step)

    _walk("R", tuple(), 0)
    return out

