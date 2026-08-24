from __future__ import annotations

from typing import Any

from ..db import connect, execute_readonly
from ..introspect import introspect
from .base import BaseDialect


class PostgresDialect(BaseDialect):
    name = "postgres"
    label = "PostgreSQL"
    supports_odoo_metadata = True

    def introspect(self) -> dict[tuple[str, str], Any]:
        with connect(self.settings) as conn:
            return introspect(conn)

    def execute_readonly(self, sql: str, max_rows: int):
        return execute_readonly(self.settings, sql, max_rows)

    def sample_values(self, schema: str, table: str, column: str, k: int = 10):
        try:
            with connect(self.settings) as conn, conn.cursor() as cur:
                q = self._qualified(schema or "public", table)
                cur.execute(
                    f"SELECT {q2(column)} AS v FROM {q} WHERE {q2(column)} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY count(*) DESC LIMIT {int(k)}",
                )
                return [str(r["v"]) for r in cur.fetchall()]
        except Exception:
            return None


def q2(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'

    def prompt_hints(self) -> str:
        return (
            "- Schema-qualify tables when the context shows a schema (e.g. public.res_partner).\n"
            "- Use ILIKE '%term%' for case-insensitive text matching.\n"
            "- Date arithmetic uses CURRENT_DATE - INTERVAL '30 days'.\n"
            "- Use ::type casts or CAST() when needed; LIMIT n is supported."
        )
