from __future__ import annotations

CANONICAL = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "pg": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",
    "mssql": "sqlserver",
    "sqlserver": "sqlserver",
    "azure-sql": "sqlserver",
    "tsql": "sqlserver",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "bq": "bigquery",
    "oracle": "oracle",
    "duckdb": "duckdb",
    "trino": "trino",
    "presto": "trino",
    "clickhouse": "clickhouse",
    "redshift": "redshift",
    "databricks": "databricks",
}


def canonical_name(raw: str | None) -> str:
    value = (raw or "postgres").lower().strip()
    canon = CANONICAL.get(value)
    if canon is None:
        known = ", ".join(sorted(set(CANONICAL.values())))
        raise ValueError(f"Unknown DB_DIALECT '{raw}'. Supported dialects: {known}")
    return canon


def get_dialect(settings):
    canon = canonical_name(getattr(settings, "db_dialect", "postgres"))

    if canon == "postgres":
        from .postgres import PostgresDialect

        return PostgresDialect(settings)
    if canon == "mysql":
        from .mysql import MySQLDialect

        return MySQLDialect(settings)
    if canon == "sqlserver":
        from .sqlserver import SQLServerDialect

        return SQLServerDialect(settings)

    from .generic import GenericSQLAlchemyDialect

    return GenericSQLAlchemyDialect(settings, scheme_key=canon)
