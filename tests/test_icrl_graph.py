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
