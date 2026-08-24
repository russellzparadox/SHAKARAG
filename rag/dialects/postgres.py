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

    def prompt_hints(self) -> str:
        return (
            "- Schema-qualify tables when the context shows a schema (e.g. public.res_partner).\n"
            "- Use ILIKE '%term%' for case-insensitive text matching.\n"
            "- Date arithmetic uses CURRENT_DATE - INTERVAL '30 days'.\n"
            "- Use ::type casts or CAST() when needed; LIMIT n is supported."
        )
