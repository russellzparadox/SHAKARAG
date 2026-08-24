from __future__ import annotations

from typing import Any

from ..introspect import Column, ForeignKey, TableRecord
from .base import BaseDialect

LABELS = {
    "snowflake": "Snowflake",
    "bigquery": "BigQuery (GoogleSQL)",
    "oracle": "Oracle",
    "duckdb": "DuckDB",
    "trino": "Trino/Presto",
    "clickhouse": "ClickHouse",
    "redshift": "Amazon Redshift",
    "databricks": "Databricks SQL",
}

SCHEME_HINTS = {
    "snowflake": (
        "- Qualify tables as DATABASE.SCHEMA.TABLE when shown.\n"
        "- ILIKE is supported; date arithmetic via DATEADD(day, -30, CURRENT_DATE())."
    ),
    "bigquery": (
        "- Qualify tables as `project.dataset.table` when shown.\n"
        "- Use SAFE_CAST and backticked identifiers; no LIMIT-less scans on large tables.\n"
        "- Date arithmetic: DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)."
    ),
    "oracle": (
        "- Fetch rows with FETCH FIRST n ROWS ONLY (12c+) instead of LIMIT.\n"
        "- No ILIKE; use UPPER(col) LIKE UPPER('%term%'). SYSDATE - 30 for dates."
    ),
    "duckdb": "- ILIKE and LIMIT are supported; rich date functions available.",
    "trino": (
        "- Use LIMIT n; ILIKE is supported.\n"
        "- Qualify as catalog.schema.table when shown; date_add('day', -30, CURRENT_DATE)."
    ),
    "clickhouse": (
        "- Use LIMIT n; case-insensitive match via positionCaseInsensitive(col, 'term') > 0 or lower().\n"
        "- Dates: today() - 30; identifiers may use backticks."
    ),
    "redshift": "- ILIKE and LIMIT are supported (Postgres-compatible); DATEDIFF(day, col, GETDATE()).",
    "databricks": (
        "- Unity Catalog names: catalog.schema.table.\n"
        "- ILIKE supported; date_sub(current_date(), 30) for dates."
    ),
}


class GenericSQLAlchemyDialect(BaseDialect):
    def __init__(self, settings, scheme_key: str) -> None:
        super().__init__(settings)
        self.scheme_key = scheme_key
        self.label = LABELS.get(scheme_key, scheme_key)

    def _url(self) -> str:
        s = self.settings
        if getattr(s, "db_url", None):
            return s.db_url
        auth = f"{s.db_user}:{s.db_password}@" if s.db_user else ""
        if self.scheme_key == "snowflake":
            return f"snowflake://{auth}{s.db_host}/{s.db_name}"
        if self.scheme_key == "bigquery":
            project = s.db_user or s.db_host
            dataset = s.db_name
            return f"bigquery://{project}/{dataset}" if project else f"bigquery:///{dataset}"
        if self.scheme_key == "duckdb":
            return f"duckdb:///{s.db_name}"
        if self.scheme_key == "redshift":
            return f"redshift+redshift_connector://{auth}{s.db_host}:{s.db_port}/{s.db_name}"
        if self.scheme_key == "clickhouse":
            return f"clickhouse+native://{auth}{s.db_host}:{s.db_port}/{s.db_name}"
        return f"{self.scheme_key}://{auth}{s.db_host}:{s.db_port}/{s.db_name}"

    def _engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise self.missing_driver("sqlalchemy") from exc
        try:
            return create_engine(self._url(), connect_args={}, future=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not create engine for {self.label}. "
                f"For warehouses set DB_URL to a full SQLAlchemy URL. Error: {exc}"
            ) from exc

    def introspect(self) -> dict[tuple[str, str], TableRecord]:
        from sqlalchemy import inspect as sa_inspect

        engine = self._engine()
        tables: dict[tuple[str, str], TableRecord] = {}
        try:
            insp = sa_inspect(engine)
            try:
                schemas = [s for s in insp.get_schema_names() if s not in ("information_schema",)]
            except Exception:
                schemas = [None]

            for schema in schemas:
                names: list[tuple[str, str]] = []
                try:
                    names += [(t, "r") for t in insp.get_table_names(schema=schema)]
                except Exception:
                    pass
                try:
                    names += [(t, "v") for t in insp.get_view_names(schema=schema)]
                except Exception:
                    pass
                for name, kind in names:
                    key = (schema, name)
                    rec = TableRecord(schema=schema or "", name=name, kind=kind, row_estimate=0, comment=None)
                    try:
                        rec.comment = insp.get_table_comment(name, schema=schema).get("text") or None
                    except Exception:
                        pass
                    try:
                        pk = insp.get_pk_constraint(name, schema=schema) or {}
                        rec.primary_key = list(pk.get("constrained_columns") or [])
                    except Exception:
                        pass
                    try:
                        for fk in insp.get_foreign_keys(name, schema=schema):
                            ref_table = fk.get("referred_table") or ""
                            ref_schema = fk.get("referred_schema")
                            fq = ref_table if not ref_schema or ref_schema == schema else f"{ref_schema}.{ref_table}"
                            obj = ForeignKey(
                                fk.get("name") or "",
                                list(fk.get("constrained_columns") or []),
                                fq,
                                list(fk.get("referred_columns") or []),
                            )
                            rec.foreign_keys.append(obj)
                            target = tables.get((ref_schema, ref_table))
                            if target is not None:
                                src = name if ref_schema == schema else f"{schema}.{name}"
                                target.referenced_by.append(
                                    ForeignKey(obj.name, obj.columns, fq, obj.ref_columns, source=src)
                                )
                    except Exception:
                        pass
                    try:
                        for ix in insp.get_indexes(name, schema=schema):
                            cols = ", ".join(ix.get("column_names") or [])
                            uniq = "UNIQUE " if ix.get("unique") else ""
                            rec.indexes.append((ix.get("name") or "", f"{uniq}INDEX ({cols})"))
                    except Exception:
                        pass
                    try:
                        for c in insp.get_columns(name, schema=schema):
                            ctype = str(c.get("type"))
                            rec.columns.append(
                                Column(
                                    name=c.get("name"),
                                    type=ctype,
                                    nullable=bool(c.get("nullable", True)),
                                    default=str(c["default"]) if c.get("default") is not None else None,
                                    comment=c.get("comment"),
                                    identity=False,
                                    generated=False,
                                    pk=c.get("name") in rec.primary_key,
                                )
                            )
                    except Exception:
                        pass
                    tables[key] = rec
        finally:
            engine.dispose()

        for rec in tables.values():
            fk_refs = {
                c: f"{fk.ref_table}({rc})"
                for fk in rec.foreign_keys
                for c, rc in zip(fk.columns, fk.ref_columns)
            }
            pk = set(rec.primary_key)
            for col in rec.columns:
                col.pk = col.name in pk
                if col.fk_ref is None:
                    col.fk_ref = fk_refs.get(col.name)

        return tables

    def execute_readonly(self, sql: str, max_rows: int):
        from sqlalchemy import text

        from ..serializers import cell

        engine = self._engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = list(result.keys())
                raw = result.fetchmany(max_rows + 1) if result.returns_rows else []
                truncated = len(raw) > max_rows
                rows = [[cell(v) for v in tuple(r)] for r in raw[:max_rows]]
                conn.rollback()
                return columns, rows, truncated
        finally:
            engine.dispose()

    def prompt_hints(self) -> str:
        hint = SCHEME_HINTS.get(self.scheme_key, "")
        base = "- Rely only on tables/columns present in the context."
        return f"{base}\n{hint}" if hint else base
