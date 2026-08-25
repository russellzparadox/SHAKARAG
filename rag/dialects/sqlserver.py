from __future__ import annotations

from typing import Any

from ..introspect import Column, ForeignKey, TableRecord
from .base import BaseDialect

TABLES_SQL = """
SELECT s.name AS ts, t.name AS tn,
       COALESCE(SUM(CASE WHEN p.index_id IN (0, 1) THEN p.rows END), 0) AS est_rows,
       MAX(CAST(ep.value AS nvarchar(4000))) AS tc
FROM sys.tables t
JOIN sys.schemas s ON s.schema_id = t.schema_id
LEFT JOIN sys.partitions p ON p.object_id = t.object_id
LEFT JOIN sys.extended_properties ep
  ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.class = 1 AND ep.name = 'MS_Description'
GROUP BY s.name, t.name
"""

VIEWS_SQL = """
SELECT s.name AS ts, v.name AS tn, OBJECT_DEFINITION(v.object_id) AS vdef
FROM sys.views v
JOIN sys.schemas s ON s.schema_id = v.schema_id
"""

COLUMNS_SQL = """
SELECT s.name AS ts, t.name AS tn, c.name AS cn, tp.name AS dtype,
       CASE WHEN tp.name IN ('char','nchar','varchar','nvarchar','binary','varbinary')
                 AND c.max_length > 0 THEN c.max_length ELSE NULL END AS maxlen,
       CASE WHEN tp.name IN ('nchar','nvarchar') AND c.max_length > 0 THEN c.max_length / 2 ELSE NULL END AS chars,
       c.precision AS prec, c.scale AS sc,
       c.is_nullable AS nul, dc.definition AS cdef,
       c.is_identity AS ident, c.is_computed AS comp,
       CAST(ep.value AS nvarchar(4000)) AS cc
FROM sys.columns c
JOIN sys.types tp ON tp.user_type_id = c.user_type_id
JOIN sys.objects t ON t.object_id = c.object_id AND t.type IN ('U', 'V')
JOIN sys.schemas s ON s.schema_id = t.schema_id
LEFT JOIN sys.default_constraints dc
  ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id
LEFT JOIN sys.extended_properties ep
  ON ep.major_id = c.object_id AND ep.minor_id = c.column_id AND ep.class = 1 AND ep.name = 'MS_Description'
ORDER BY s.name, t.name, c.column_id
"""

KEYS_SQL = """
SELECT s.name AS ts, t.name AS tn, kcu.constraint_name AS cn, tc.constraint_type AS ctype,
       kcu.column_name AS col
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_schema = tc.constraint_schema
 AND kcu.constraint_name = tc.constraint_name
 AND kcu.table_name = tc.table_name
JOIN sys.schemas s ON s.name = tc.constraint_schema
JOIN sys.tables t ON t.name = tc.table_name
WHERE tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
ORDER BY s.name, t.name, tc.constraint_name, kcu.ordinal_position
"""

FKS_SQL = """
SELECT s.name AS ts, t.name AS tn, col.name AS col, fk.name AS cn,
       rs.name AS rts, rt.name AS rtn, rcol.name AS rcol
FROM sys.foreign_keys fk
JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
JOIN sys.tables t ON t.object_id = fk.parent_object_id
JOIN sys.schemas s ON s.schema_id = t.schema_id
JOIN sys.columns col ON col.object_id = t.object_id AND col.column_id = fkc.parent_column_id
JOIN sys.tables rt ON rt.object_id = fk.referenced_object_id
JOIN sys.schemas rs ON rs.schema_id = rt.schema_id
JOIN sys.columns rcol ON rcol.object_id = rt.object_id AND rcol.column_id = fkc.referenced_column_id
ORDER BY s.name, t.name, fk.name, fkc.constraint_column_id
"""

INDEXES_SQL = """
SELECT s.name AS ts, t.name AS tn, i.name AS ix, i.is_unique AS uniq,
       STUFF((SELECT ',' + c2.name
              FROM sys.index_columns ic2
              JOIN sys.columns c2 ON c2.object_id = ic2.object_id AND c2.column_id = ic2.column_id
              WHERE ic2.object_id = i.object_id AND ic2.index_id = i.index_id AND ic2.is_included_column = 0
              ORDER BY ic2.key_ordinal
              FOR XML PATH('')), 1, 1, '') AS cols
FROM sys.indexes i
JOIN sys.tables t ON t.object_id = i.object_id
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE i.name IS NOT NULL AND i.is_primary_key = 0
"""


def _mssql_type(dtype: str, maxlen, chars, prec, sc) -> str:
    if dtype in ("char", "varchar", "binary", "varbinary") and maxlen:
        return f"{dtype}({maxlen})"
    if dtype in ("nchar", "nvarchar") and chars:
        return f"{dtype}({chars})"
    if dtype == "decimal" or dtype == "numeric":
        return f"{dtype}({prec},{sc})"
    return dtype


def build_mssql_catalog(
    tables_rows: list[dict],
    view_rows: list[dict],
    column_rows: list[dict],
    key_rows: list[dict],
    fk_rows: list[dict],
    index_rows: list[dict],
) -> dict[tuple[str, str], TableRecord]:
    tables: dict[tuple[str, str], TableRecord] = {}

    for row in tables_rows:
        key = (row["ts"], row["tn"])
        tables[key] = TableRecord(
            schema=row["ts"],
            name=row["tn"],
            kind="r",
            row_estimate=int(row["est_rows"] or 0),
            comment=(row["tc"] or "").strip() or None,
        )

    for row in view_rows:
        key = (row["ts"], row["tn"])
        rec = TableRecord(
            schema=row["ts"], name=row["tn"], kind="v", row_estimate=0, comment=None
        )
        rec.view_def = row["vdef"]
        tables[key] = rec

    for row in column_rows:
        rec = tables.get((row["ts"], row["tn"]))
        if rec is None:
            continue
        default = row["cdef"]
        rec.columns.append(
            Column(
                name=row["cn"],
                type=_mssql_type(row["dtype"], row["maxlen"], row["chars"], row["prec"], row["sc"]),
                nullable=bool(row["nul"]),
                default=default,
                comment=(row["cc"] or "").strip() or None,
                identity=bool(row["ident"]),
                generated=bool(row["comp"]),
            )
        )

    grouped_keys: dict[tuple[str, str, str], dict] = {}
    for row in key_rows:
        entry = grouped_keys.setdefault(
            (row["ts"], row["tn"], row["cn"]), {"type": row["ctype"], "columns": []}
        )
        if row["col"] not in entry["columns"]:
            entry["columns"].append(row["col"])

    for (schema, table, cname), entry in grouped_keys.items():
        rec = tables.get((schema, table))
        if rec is None:
            continue
        if entry["type"] == "PRIMARY KEY":
            rec.primary_key = entry["columns"]
        else:
            rec.unique_constraints.append(entry["columns"])

    grouped_fks: dict[tuple[str, str, str], dict] = {}
    for row in fk_rows:
        entry = grouped_fks.setdefault(
            (row["ts"], row["tn"], row["cn"]),
            {"columns": [], "rcols": [], "rts": row["rts"], "rtn": row["rtn"]},
        )
        if row["col"] not in entry["columns"]:
            entry["columns"].append(row["col"])
        entry["rcols"].append(row["rcol"])

    for (schema, table, cname), entry in grouped_fks.items():
        rec = tables.get((schema, table))
        if rec is None:
            continue
        ref = entry["rtn"] if entry["rts"] == schema else f"{entry['rts']}.{entry['rtn']}"
        fk = ForeignKey(cname, entry["columns"], ref, entry["rcols"])
        rec.foreign_keys.append(fk)
        target = tables.get((entry["rts"], entry["rtn"]))
        if target is not None:
            src = table if schema == target.schema else f"{schema}.{table}"
            target.referenced_by.append(
                ForeignKey(cname, fk.columns, ref, fk.ref_columns, source=src)
            )

    for row in index_rows:
        rec = tables.get((row["ts"], row["tn"]))
        if rec is not None and row["cols"]:
            uniq = "UNIQUE " if row["uniq"] else ""
            rec.indexes.append((row["ix"], f"{uniq}INDEX ({row['cols']})"))

    for rec in tables.values():
        pk = set(rec.primary_key)
        fk_refs = {
            c: f"{fk.ref_table}({rc})"
            for fk in rec.foreign_keys
            for c, rc in zip(fk.columns, fk.ref_columns)
        }
        for col in rec.columns:
            col.pk = col.name in pk
            col.fk_ref = fk_refs.get(col.name)

    return tables


class SQLServerDialect(BaseDialect):
    name = "sqlserver"
    label = "SQL Server (T-SQL)"

    def _connect(self, query_timeout: int | None = None):
        try:
            import pymssql
        except ImportError as exc:
            raise self.missing_driver("pymssql") from exc
        s = self.settings
        if query_timeout is None:
            query_timeout = max(int(s.statement_timeout_ms / 1000), 1)
        return pymssql.connect(
            server=s.db_host,
            port=str(s.db_port),
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
            login_timeout=10,
            timeout=query_timeout,
            as_dict=True,
        )

    def introspect(self) -> dict[tuple[str, str], TableRecord]:
        # Catalog queries over large warehouses can run well past the per-query
        # statement timeout — allow up to 5 minutes for introspection.
        conn = self._connect(query_timeout=300)
        try:
            with conn.cursor() as cur:
                cur.execute(TABLES_SQL)
                tables_rows = cur.fetchall()
                cur.execute(VIEWS_SQL)
                view_rows = cur.fetchall()
                cur.execute(COLUMNS_SQL)
                column_rows = cur.fetchall()
                cur.execute(KEYS_SQL)
                key_rows = cur.fetchall()
                cur.execute(FKS_SQL)
                fk_rows = cur.fetchall()
                cur.execute(INDEXES_SQL)
                index_rows = cur.fetchall()
        finally:
            conn.close()
        return build_mssql_catalog(tables_rows, view_rows, column_rows, key_rows, fk_rows, index_rows)

    def sample_values(self, schema: str, table: str, column: str, k: int = 10):
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                q = self._qualified(schema or "dbo", table)
                cur.execute(
                    f"SELECT TOP {int(k)} [{column.replace(']', ']]')}] AS v FROM {q} "
                    f"WHERE [{column.replace(']', ']]')}] IS NOT NULL "
                    f"GROUP BY [{column.replace(']', ']]')}] ORDER BY COUNT(*) DESC"
                )
                return [str(r["v"]) for r in cur.fetchall()]
        except Exception:
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def execute_readonly(self, sql: str, max_rows: int):
        from ..serializers import row_to_list

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN TRANSACTION")
                try:
                    cur.execute(sql)
                    columns = [d[0] for d in cur.description] if cur.description else []
                    raw = cur.fetchmany(max_rows + 1)
                    truncated = len(raw) > max_rows
                    raw = raw[:max_rows]
                    rows = [row_to_list(r) for r in raw]
                finally:
                    cur.execute("ROLLBACK")
                return columns, rows, truncated
        finally:
            conn.close()

    def prompt_hints(self) -> str:
        return (
            "- Quote identifiers with [brackets] when needed.\n"
            "- Use TOP (n) or OFFSET n ROWS FETCH NEXT m ROWS ONLY instead of LIMIT.\n"
            "- There is no ILIKE; use LOWER(col) LIKE '%term%'.\n"
            "- Date functions: GETDATE(), DATEADD(day, -30, GETDATE()); string concat uses + or CONCAT()."
        )
