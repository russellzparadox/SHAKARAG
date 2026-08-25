from __future__ import annotations

import re

from .introspect import TableRecord

FACT_NAME = re.compile(r"(^|_)(fact|facts|f_|ftbl|measure)", re.IGNORECASE)
DIM_NAME = re.compile(r"(^|_)(dim|dimension|dm_)|(_dim$)", re.IGNORECASE)
BRIDGE_NAME = re.compile(r"(bridge|junction|mapping|_link_|_xref)", re.IGNORECASE)
RELATION_NAME = re.compile(r"_rel$|_relations?$", re.IGNORECASE)

MEASURE_WORDS = (
    "amount", "qty", "quantity", "price", "total", "revenue", "sales", "cost",
    "margin", "profit", "balance", "sum", "net", "gross", "tax_amount", "fee",
    "discount_amount", "weight", "volume", "duration", "count_", "_count",
    "amount_total", "price_unit", "debit", "credit",
)

DATE_WORDS = ("date", "_at", "timestamp", "_on", "day", "month", "quarter", "year", "week")

NUMERIC_TYPES = ("decimal", "numeric", "float", "double", "real", "money", "int", "integer", "bigint", "smallint", "number")
TEXT_TYPES = ("char", "text", "string", "varchar", "enum", "nvarchar")

SECRET_WORDS = ("password", "passwd", "secret", "token", "hash", "api_key", "apikey", "private_key", "credential")


def _is_numeric_type(t: str) -> bool:
    t = t.lower()
    return any(t.startswith(nt) for nt in NUMERIC_TYPES)


def _is_textual_type(t: str) -> bool:
    t = t.lower()
    return any(t.startswith(tt) for tt in TEXT_TYPES)


def _measures(rec: TableRecord) -> list[str]:
    out = []
    for col in rec.columns:
        if not _is_numeric_type(col.type):
            continue
        n = col.name.lower()
        if any(w in n for w in MEASURE_WORDS):
            out.append(col.name)
    return out


DATE_TYPES = ("date", "timestamp", "datetime", "time")


def _date_like(rec: TableRecord) -> list[str]:
    out = []
    for c in rec.columns:
        n = c.name.lower()
        t = c.type.lower()
        is_date_type = any(t.startswith(dt) for dt in DATE_TYPES)
        if n == "id":
            continue
        if is_date_type or (any(w in n for w in DATE_WORDS) and not _is_textual_type(t)):
            out.append(c.name)
    return out


def classify_table(rec: TableRecord, median_rows: float = 0.0) -> None:
    name = rec.name
    ncols = len(rec.columns)
    n_fks = len(rec.foreign_keys)
    n_refs = len(rec.referenced_by)
    measures = _measures(rec)
    dates = [c for c in _date_like(rec)]
    nonkey_cols = [
        c for c in rec.columns
        if not c.pk and c.fk_ref is None and c.name.lower() not in ("id",)
    ]

    if FACT_NAME.search(name):
        rec.warehouse_role = "fact"
        rec.role_reason = "name suggests fact table"
        return
    if DIM_NAME.search(name):
        rec.warehouse_role = "dimension"
        rec.role_reason = "name suggests dimension table"
        return
    if RELATION_NAME.search(name) or BRIDGE_NAME.search(name):
        if n_fks >= 2:
            rec.warehouse_role = "relation"
            rec.role_reason = f"join/bridge table linking {n_fks} tables"
            return

    keyish = sum(1 for c in rec.columns if c.pk or c.fk_ref is not None or c.name.lower() == "id")
    if n_fks >= 2 and ncols <= max(5, keyish + 1) and not measures:
        rec.warehouse_role = "relation"
        rec.role_reason = f"thin join table with {n_fks} foreign keys"
        return

    bigger_than_dims = median_rows > 0 and rec.row_estimate > median_rows * 3
    fact_score = (
        n_fks * 1.5
        + len(measures) * 1.2
        + min(len(dates), 3) * 0.4
        + (2.0 if bigger_than_dims else 0.0)
    )
    dim_score = (
        max(0, 2 - n_fks) * 0.8
        + n_refs * 0.9
        + len(nonkey_cols) * 0.15
        + (1.5 if n_fks <= 1 else 0.0)
    )

    if fact_score >= 3.0 and fact_score > dim_score:
        role = "fact"
        why = []
        if n_fks:
            why.append(f"{n_fks} outgoing FKs")
        if measures:
            why.append(f"measures: {', '.join(measures[:3])}")
        if bigger_than_dims:
            why.append("much larger than typical dimension")
        rec.role_reason = "; ".join(why) or "structure resembles a fact table"
    elif dim_score >= 1.6:
        role = "dimension"
        rec.role_reason = f"descriptive attributes, referenced by {n_refs} table(s)"
    else:
        role = "unknown"
        rec.role_reason = ""
    rec.warehouse_role = role


def classify_tables(tables: list[TableRecord]) -> dict[str, int]:
    row_estimates = sorted(max(t.row_estimate, 0) for t in tables if t.kind == "TABLE")
    median = row_estimates[len(row_estimates) // 2] if row_estimates else 0.0
    for rec in tables:
        if rec.kind in ("r", "p"):
            classify_table(rec, median_rows=float(median))
    counts: dict[str, int] = {}
    for rec in tables:
        counts[rec.warehouse_role] = counts.get(rec.warehouse_role, 0) + 1
    return counts


def candidate_value_columns(rec: TableRecord, max_per_table: int = 8) -> list[str]:
    if rec.kind not in ("r", "p"):
        return []
    picked: list[str] = []
    for col in rec.columns:
        lname = col.name.lower()
        if col.pk or col.fk_ref or col.identity or col.generated:
            continue
        if any(sw in lname for sw in SECRET_WORDS):
            continue
        if not _is_textual_type(col.type):
            continue
        if len(lname) < 3:
            continue
        picked.append(col.name)
        if len(picked) >= max_per_table:
            break
    return picked


DW_PROMPT_BLOCK = """- Data-warehouse guidance: tables marked FACT hold transaction rows/measures; DIMENSION tables hold descriptive attributes.
  Aggregate measures on the fact table, put filters on joined dimension columns, and join strictly via the shown FKs.
  Respect the stated grain of the fact before aggregating; prefer SUM/COUNT on facts over pulling raw rows.
  Join tables marked relation/bridge connect many-to-many pairs.
- When the user asks "how many X are there" or wants a list of entities, count/list the DIMENSION table named after that entity (e.g. DimCompany for "companies") — never an ETL/staging/audit view.
- When the user names a specific item ("info of Xperial", "orders of Acme"), find the DIMENSION whose name matches the entity word (supplier→*Supplier*, customer→*Customer*), and filter its human-readable columns (Title, Name, Description, Code) with LIKE '%stem%' using a word stem of the item, trying alternate stems if needed. Do not switch to unrelated tables."""


def warehouse_hints_block() -> str:
    return DW_PROMPT_BLOCK
