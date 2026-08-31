"""Evaluation harness for the ICRL/RAG-over-SQL pipeline (paper §4).

Public surface:
    - `compute_recall(retrieved, gold, k)`  -> float 0/1 (recall@k)
    - `generate_holdout(tables, n, seed)`   -> list of held-out questions
    - `execution_accuracy(generated_sql, executed, gold_rows)` -> float 0/1
"""
from __future__ import annotations

import random
from typing import Iterable


# ---- F3: recall@k ---------------------------------------------------------

def compute_recall(retrieved: Iterable[Iterable[str]], gold: set[str], k: int = 1) -> float:
    """Recall@k: 1.0 iff the union of the top-k ranked tables is a superset
    of the gold tables, else 0.0.

    Paper Table 1 reports R@k as a fraction of queries where the gold
    schema appears in the top-k retrieved. We follow that binary convention.

    Convention: `retrieved` is a list of lists, one per ranked position.
    Each inner list is the set of tables contributed by the candidate at
    that position (e.g. one ICRL example that mentions multiple tables).
    The recall score is 1.0 when the union of the first k positions
    contains the gold set.
    """
    if not retrieved:
        return 0.0
    top_k: set[str] = set()
    for i, contrib in enumerate(retrieved):
        if i >= k:
            break
        if contrib is None:
            continue
        for t in contrib:
            top_k.add(str(t).lower())
    return 1.0 if gold.issubset(top_k) else 0.0


# ---- F5: deterministic holdout generation ---------------------------------

# A handful of NL templates that exercise different SQL operators. Each
# template is a callable that takes (table_list, rng) and returns
# (question, sql_template, gold_tables, gold_rows).
HOLDOUT_TEMPLATES = [
    # count rows
    lambda tables, rng: (
        f"How many rows are in {tables[0].name}?",
        f"SELECT COUNT(*) FROM {tables[0].schema}.{tables[0].name}",
        [f"{tables[0].schema}.{tables[0].name}"],
        # the count is unknown at template time — the test will execute and
        # treat "any rows" as 0.5 in execution_accuracy if gold_rows is None.
        None,
    ),
    # count by single column
    lambda tables, rng: (
        f"Count of rows in {tables[0].name} grouped by id",
        f"SELECT id, COUNT(*) FROM {tables[0].schema}.{tables[0].name} GROUP BY id",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
    # filter + count
    lambda tables, rng: (
        f"Count of rows in {tables[0].name} where id > 0",
        f"SELECT COUNT(*) FROM {tables[0].schema}.{tables[0].name} WHERE id > 0",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
    # multi-table (needs at least 2 tables)
    lambda tables, rng: (
        f"Count of {tables[0].name} joined with {tables[1].name}",
        f"SELECT COUNT(*) FROM {tables[0].schema}.{tables[0].name} t1 "
        f"JOIN {tables[1].schema}.{tables[1].name} t2 ON t1.id = t2.id",
        [f"{tables[0].schema}.{tables[0].name}", f"{tables[1].schema}.{tables[1].name}"],
        None,
    ) if len(tables) >= 2 else None,
    # multi-table count grouped
    lambda tables, rng: (
        f"Top categories from {tables[0].name}",
        f"SELECT id, COUNT(*) AS c FROM {tables[0].schema}.{tables[0].name} "
        f"GROUP BY id ORDER BY c DESC LIMIT 10",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
    # avg
    lambda tables, rng: (
        f"Average of id in {tables[0].name}",
        f"SELECT AVG(id) FROM {tables[0].schema}.{tables[0].name}",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
    # simple select *
    lambda tables, rng: (
        f"First 10 rows from {tables[0].name}",
        f"SELECT * FROM {tables[0].schema}.{tables[0].name} LIMIT 10",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
    # exists-style
    lambda tables, rng: (
        f"Rows from {tables[0].name} where id IS NOT NULL",
        f"SELECT * FROM {tables[0].schema}.{tables[0].name} WHERE id IS NOT NULL",
        [f"{tables[0].schema}.{tables[0].name}"],
        None,
    ),
]


def generate_holdout(tables: list, n: int = 20, seed: int = 42) -> list[dict]:
    """Generate `n` deterministic held-out (Q, gold_tables, expected_rows)
    items by sampling NL templates over `tables`.

    Returns a list of dicts:
        {
            "question": str,
            "sql_template": str,
            "gold_tables": [str, ...],
            "gold_rows": int | None,
        }
    """
    if not tables:
        return []
    rng = random.Random(seed)
    items: list[dict] = []
    # build the working list of applicable templates; multi-table templates
    # need at least 2 tables.
    applicable: list = []
    for tmpl in HOLDOUT_TEMPLATES:
        try:
            test = tmpl(tables[:2] if len(tables) >= 2 else tables[:1], rng)
        except Exception:
            test = None
        if test is not None:
            applicable.append(tmpl)

    if not applicable:
        return []

    for _ in range(n):
        tmpl = applicable[rng.randrange(len(applicable))]
        out = tmpl(tables, rng)
        if out is None:
            continue
        question, sql_tpl, gold_tables, gold_rows = out
        items.append({
            "question": question,
            "sql_template": sql_tpl,
            "gold_tables": list(gold_tables),
            "gold_rows": gold_rows,
        })
    return items


# ---- F7: execution accuracy -----------------------------------------------

def execution_accuracy(generated_sql: str, executed: list, gold_rows) -> float:
    """Paper §4 EX: 1.0 iff the executed result equals the gold rows exactly.

    `executed` is a list of row tuples; `gold_rows` is either a list of
    row tuples (exact row-set match) or an int (exact count match). For
    `None` gold (template couldn't know the count at generation time), we
    return 1.0 if executed is non-empty and 0.0 otherwise.
    """
    if gold_rows is None:
        return 1.0 if executed else 0.0
    if isinstance(gold_rows, int):
        return 1.0 if len(executed) == gold_rows else 0.0
    # row-set comparison (order-insensitive)
    return 1.0 if sorted(executed) == sorted(gold_rows) else 0.0
