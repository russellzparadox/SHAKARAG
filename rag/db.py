from __future__ import annotations

import datetime as dt
import decimal
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=10,
        autocommit=True,
        row_factory=dict_row,
    )


def fetch_all(settings: Settings, sql: str) -> list[dict[str, Any]]:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def _cell(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, memoryview)):
        return repr(bytes(value))[:120]
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def execute_readonly(
    settings: Settings, sql: str, max_rows: int
) -> tuple[list[str], list[list[Any]], bool]:
    conn = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=10,
        autocommit=True,
        options=f"-c statement_timeout={settings.statement_timeout_ms}",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN TRANSACTION READ ONLY")
            cur.execute(sql)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            cur.execute("ROLLBACK")
            return (
                columns,
                [[_cell(v) for v in row] for row in rows],
                truncated,
            )
    finally:
        conn.close()
