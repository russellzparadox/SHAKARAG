from __future__ import annotations

import psycopg
from dataclasses import dataclass


@dataclass
class ModelInfo:
    model: str
    label: str | None
    info: str | None


@dataclass
class FieldInfo:
    description: str | None
    help: str | None
    ttype: str | None
    relation: str | None


MODELS_SQL = "SELECT model, name, COALESCE(info, '') AS info FROM ir_model"
FIELDS_SQL = """
SELECT model, name, field_description, help, ttype, relation
FROM ir_model_fields
"""


def _clean(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = (
            value.get("en_US")
            or next((v for v in value.values() if v), None)
            or ""
        )
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def fetch_odoo_metadata(
    conn: psycopg.Connection,
) -> tuple[dict[str, ModelInfo], dict[tuple[str, str], FieldInfo]]:
    models_by_table: dict[str, ModelInfo] = {}
    fields_by_key: dict[tuple[str, str], FieldInfo] = {}

    try:
        with conn.cursor() as cur:
            cur.execute(MODELS_SQL)
            for row in cur.fetchall():
                table = row["model"].replace(".", "_")
                models_by_table[table] = ModelInfo(
                    model=row["model"],
                    label=_clean(row["name"]),
                    info=_clean(row["info"]),
                )

            cur.execute(FIELDS_SQL)
            for row in cur.fetchall():
                table = row["model"].replace(".", "_")
                fields_by_key[(table, row["name"])] = FieldInfo(
                    description=_clean(row["field_description"]),
                    help=_clean(row["help"]),
                    ttype=row["ttype"],
                    relation=row["relation"],
                )
    except psycopg.errors.UndefinedTable:
        pass

    return models_by_table, fields_by_key
