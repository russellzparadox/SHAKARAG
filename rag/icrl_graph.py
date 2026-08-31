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

# Implementation is added task-by-task in TDD; this docstring + the public
# surface above are committed first per Phase A.1.
