from __future__ import annotations

from typing import Any

from ..introspect import Column, ForeignKey, TableRecord
from .base import BaseDialect

EXCLUDED_SCHEMAS = ("mysql", "sys", "information_schema", "performance_schema")

TABLES_SQL = """
SELECT table_schema AS ts, table_name AS tn, table_type AS tt,
       COALESCE(table_rows, 0) AS est_rows,
       NULLIF(table_comment, '') AS tc
FROM information_schema.tables
WHERE table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
"""

COLUMNS_SQL = """
SELECT table_schema AS ts, table_name AS tn, column_name AS cn, column_type AS ct,
       is_nullable AS nul, column_default AS cd, column_comment AS cc, extra AS ex
FROM information_schema.columns
WHERE table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
ORDER BY table_schema, table_name, ordinal_position
"""

KEYS_SQL = """
SELECT tc.table_schema AS ts, tc.table_name AS tn, tc.constraint_name AS cn,
       tc.constraint_type AS ctype, kcu.column_name AS col
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON kcu.constraint_schema = tc.constraint_schema
 AND kcu.constraint_name = tc.constraint_name
 AND kcu.table_name = tc.table_name
WHERE tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
  AND tc.table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
"""

FKS_SQL = """
SELECT constraint_schema AS ts, table_name AS tn, column_name AS col, constraint_name AS cn,
       referenced_table_schema AS rts, referenced_table_name AS rtn, referenced_column_name AS rcol
FROM information_schema.key_column_usage
WHERE referenced_table_name IS NOT NULL
  AND constraint_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
ORDER BY constraint_schema, table_name, constraint_name, ordinal_position
"""

INDEXES_SQL = """
SELECT table_schema AS ts, table_name AS tn, index_name AS ix,
       GROUP_CONCAT(column_name ORDER BY seq_in_index) AS cols,
       non_unique AS nu
FROM information_schema.statistics
WHERE table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
  AND index_name <> 'PRIMARY'
GROUP BY table_schema, table_name, index_name, nu
"""

VIEWDEFS_SQL = """
SELECT table_schema AS ts, table_name AS tn, view_definition AS vdef
FROM information_schema.views
WHERE table_schema NOT IN ('mysql', 'sys', 'information_schema', 'performance_schema')
"""


def build_mysql_catalog(
    tables_rows: list[dict],
    column_rows: list[dict],
    key_rows: list[dict],
    fk_rows: list[dict],
    index_rows: list[dict],
    viewdef_rows: list[dict],
) -> dict[tuple[str, str], TableRecord]:
    tables: dict[tuple[str, str], TableRecord] = {}

    for row in tables_rows:
        key = (row["ts"], row["tn"])
        kind = "v" if row.get("tt") == "VIEW" else "r"
        tables[key] = TableRecord(
            schema=row["ts"],
            name=row["tn"],
            kind=kind,
            row_estimate=int(row["est_rows"] or 0),
            comment=row["tc"],
        )

    for row in column_rows:
        rec = tables.get((row["ts"], row["tn"]))
        if rec is None:
            continue
        extra = (row["ex"] or "").lower()
        default = row["cd"]
        rec.columns.append(
            Column(
                name=row["cn"],
                type=row["ct"],
                nullable=row["nul"] == "YES",
                default=str(default) if default is not None else None,
                comment=(row["cc"] or "").strip() or None,
                identity="auto_increment" in extra,
                generated="generated" in extra or "virtual" in extra,
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
            {"columns": [], "ref": [], "rts": row["rts"], "rtn": row["rtn"]},
        )
        if row["col"] not in entry["columns"]:
            entry["columns"].append(row["col"])
        fq = row["rtn"] if row["rts"] == row["ts"] else f"{row['rts']}.{row['rtn']}"
        entry["ref"] = fq
        entry.setdefault("rcols", []).append(row["rcol"])

    for (schema, table, cname), entry in grouped_fks.items():
        rec = tables.get((schema, table))
        if rec is None:
            continue
        fk = ForeignKey(cname, entry["columns"], entry["ref"], entry.get("rcols", []))
        rec.foreign_keys.append(fk)
        target_key = (entry["rts"], entry["rtn"])
        target = tables.get(target_key)
        if target is not None:
            src = table if schema == target_key[0] else f"{schema}.{table}"
            target.referenced_by.append(
                ForeignKey(cname, fk.columns, entry["ref"], fk.ref_columns, source=src)
            )

    for row in index_rows:
        rec = tables.get((row["ts"], row["tn"]))
        if rec is not None and row["ix"]:
            uniq = "" if row["nu"] else "UNIQUE "
            rec.indexes.append((row["ix"], f"{uniq}INDEX ({row['cols']})"))

    for row in viewdef_rows:
        rec = tables.get((row["ts"], row["tn"]))
        if rec is not None:
            rec.view_def = row["vdef"]

    pk_sets = {k: set(rec.primary_key) for k, rec in tables.items()}
    fk_cols: dict[tuple[str, str], dict[str, str]] = {}
    for rec in tables.values():
        for fk in rec.foreign_keys:
            for c, rc in zip(fk.columns, fk.ref_columns):
                fk_cols.setdefault((rec.schema, rec.name), {})[c] = f"{fk.ref_table}({rc})"
    for key, rec in tables.items():
        refs = fk_cols.get(key, {})
        pks = pk_sets[key]
        for col in rec.columns:
            col.pk = col.name in pks
            col.fk_ref = refs.get(col.name)

    return tables


class MySQLDialect(BaseDialect):
    name = "mysql"
    label = "MySQL/MariaDB"

    def _connect(self):
        try:
            import pymysql
        except ImportError as exc:
            raise self.missing_driver("pymysql") from exc
        s = self.settings
        return pymysql.connect(
            host=s.db_host,
            port=s.db_port,
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _fetch_all(self, cur, sql: str) -> list[dict]:
        cur.execute(sql)
        return cur.fetchall()

    def introspect(self) -> dict[tuple[str, str], TableRecord]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                tables_rows = self._fetch_all(cur, TABLES_SQL)
                column_rows = self._fetch_all(cur, COLUMNS_SQL)
                key_rows = self._fetch_all(cur, KEYS_SQL)
                fk_rows = self._fetch_all(cur, FKS_SQL)
                index_rows = self._fetch_all(cur, INDEXES_SQL)
                viewdef_rows = self._fetch_all(cur, VIEWDEFS_SQL)
        finally:
            conn.close()
        return build_mysql_catalog(tables_rows, column_rows, key_rows, fk_rows, index_rows, viewdef_rows)

    def execute_readonly(self, sql: str, max_rows: int):
        import pymysql

        from ..serializers import row_to_list

        s = self.settings
        conn = pymysql.connect(
            host=s.db_host,
            port=s.db_port,
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
            charset="utf8mb4",
            autocommit=False,
            read_timeout=max(int(s.statement_timeout_ms / 1000), 1),
            connect_timeout=10,
        )
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute("SET SESSION TRANSACTION READ ONLY = 1")
                except Exception:
                    pass
                cur.execute("START TRANSACTION READ ONLY")
                cur.execute(sql)
                columns = [d[0] for d in cur.description] if cur.description else []
                raw = cur.fetchmany(max_rows + 1)
                truncated = len(raw) > max_rows
                raw = raw[:max_rows]
                rows = [row_to_list(r) for r in raw]
                cur.execute("ROLLBACK")
                return columns, rows, truncated
        finally:
            conn.close()

    def sample_values(self, schema: str, table: str, column: str, k: int = 10):
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                q = self._qualified(schema or self.settings.db_name, table, "`")
                cur.execute(
                    f"SELECT `{column.replace('`', '')}` AS v FROM {q} "
                    f"WHERE `{column.replace('`', '')}` IS NOT NULL "
                    f"GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {int(k)}"
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

    def prompt_hints(self) -> str:
        return (
            "- Quote identifiers with backticks when needed.\n"
            "- There is no ILIKE; use LOWER(col) LIKE '%term%' (or col LIKE on MySQL where collations are case-insensitive by default).\n"
            "- Date arithmetic uses CURDATE() - INTERVAL 30 DAY or DATE_SUB().\n"
            "- LIMIT n is supported. JSON access uses JSON_EXTRACT()/->> style operators."
        )
