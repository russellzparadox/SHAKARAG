from __future__ import annotations

from dataclasses import dataclass, field

from psycopg.rows import dict_row

EXCLUDED_SCHEMAS = ("pg_catalog", "information_schema")

COLUMNS_SQL = """
SELECT n.nspname AS schema,
       c.relname AS table_name,
       c.relkind AS kind,
       GREATEST(c.reltuples, 0)::bigint AS row_estimate,
       obj_description(c.oid, 'pg_class') AS comment,
       a.attnum,
       a.attname AS column_name,
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
       NOT a.attnotnull AS is_nullable,
       pg_get_expr(d.adbin, d.adrelid) AS column_default,
       col_description(a.attrelid, a.attnum) AS col_comment,
       a.attidentity <> '' AS is_identity,
       a.attgenerated <> '' AS is_generated
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg\\_%'
  AND c.relkind IN ('r', 'v', 'm', 'p')
ORDER BY n.nspname, c.relname, a.attnum
"""

CONSTRAINTS_SQL = """
SELECT n.nspname AS schema,
       cl.relname AS table_name,
       con.conname AS constraint_name,
       con.contype AS constraint_type,
       ck.attnum AS col_attnum,
       fk.attnum AS ref_attnum,
       rn.nspname AS ref_schema,
       rcl.relname AS ref_table
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class cl ON cl.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = cl.relnamespace
LEFT JOIN pg_catalog.pg_class rcl ON rcl.oid = con.confrelid
LEFT JOIN pg_catalog.pg_namespace rn ON rn.oid = rcl.relnamespace
CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS ck(attnum, ord)
LEFT JOIN LATERAL unnest(coalesce(con.confkey, '{}')) WITH ORDINALITY AS fk(attnum, ord)
  ON fk.ord = ck.ord
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg\\_%'
  AND con.contype IN ('p', 'f', 'u')
ORDER BY n.nspname, cl.relname, con.conname, ck.ord
"""

INDEXES_SQL = """
SELECT schemaname, tablename, indexname, indexdef
FROM pg_catalog.pg_indexes
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND schemaname NOT LIKE 'pg\\_%'
ORDER BY schemaname, tablename, indexname
"""

VIEWDEFS_SQL = """
SELECT n.nspname AS schema, c.relname AS table_name, pg_get_viewdef(c.oid, true) AS view_def
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg\\_%'
  AND c.relkind IN ('v', 'm')
"""


@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    default: str | None
    comment: str | None
    identity: bool
    generated: bool
    pk: bool = False
    fk_ref: str | None = None
    sample_values: list[str] | None = None


@dataclass
class ForeignKey:
    name: str
    columns: list[str]
    ref_table: str
    ref_columns: list[str]
    source: str | None = None


@dataclass
class TableRecord:
    schema: str
    name: str
    kind: str
    row_estimate: int
    comment: str | None
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    referenced_by: list[ForeignKey] = field(default_factory=list)
    unique_constraints: list[list[str]] = field(default_factory=list)
    indexes: list[tuple[str, str]] = field(default_factory=list)
    view_def: str | None = None
    warehouse_role: str = "unknown"
    role_reason: str = ""

    @property
    def qualified(self) -> str:
        if self.schema == "public":
            return self.name
        return f"{self.schema}.{self.name}"

    @property
    def kind_label(self) -> str:
        return {"r": "TABLE", "p": "PARTITIONED TABLE", "v": "VIEW", "m": "MATERIALIZED VIEW"}[
            self.kind
        ]


def _key(schema: str, table: str) -> tuple[str, str]:
    return schema, table


def introspect(conn: psycopg.Connection) -> dict[tuple[str, str], TableRecord]:
    tables: dict[tuple[str, str], TableRecord] = {}
    attnum_map: dict[tuple[str, str], dict[int, str]] = {}

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(COLUMNS_SQL)
        for row in cur.fetchall():
            key = _key(row["schema"], row["table_name"])
            rec = tables.get(key)
            if rec is None:
                rec = TableRecord(
                    schema=row["schema"],
                    name=row["table_name"],
                    kind=row["kind"],
                    row_estimate=row["row_estimate"] or 0,
                    comment=(row["comment"] or "").strip() or None,
                )
                tables[key] = rec
                attnum_map[key] = {}
            attnum_map[key][row["attnum"]] = row["column_name"]
            rec.columns.append(
                Column(
                    name=row["column_name"],
                    type=row["data_type"],
                    nullable=row["is_nullable"],
                    default=row["column_default"],
                    comment=(row["col_comment"] or "").strip() or None,
                    identity=row["is_identity"],
                    generated=row["is_generated"],
                )
            )

        cur.execute(CONSTRAINTS_SQL)
        grouped: dict[tuple[str, str, str], dict] = {}
        for row in cur.fetchall():
            key = (row["schema"], row["table_name"], row["constraint_name"])
            entry = grouped.setdefault(
                key,
                {
                    "type": row["constraint_type"],
                    "columns": [],
                    "ref_columns": [],
                    "ref_schema": row["ref_schema"],
                    "ref_table": row["ref_table"],
                },
            )
            table_key = (row["schema"], row["table_name"])
            col_name = attnum_map.get(table_key, {}).get(row["col_attnum"])
            if col_name and col_name not in entry["columns"]:
                entry["columns"].append(col_name)
            if entry["type"] == "f" and row["ref_attnum"]:
                ref_key = (entry["ref_schema"], entry["ref_table"])
                ref_col = attnum_map.get(ref_key, {}).get(row["ref_attnum"])
                if ref_col and ref_col not in entry["ref_columns"]:
                    entry["ref_columns"].append(ref_col)

        for (schema, table, cname), entry in grouped.items():
            rec = tables[(schema, table)]
            if entry["type"] == "p":
                rec.primary_key = entry["columns"]
            elif entry["type"] == "u":
                rec.unique_constraints.append(entry["columns"])
            elif entry["type"] == "f" and entry["ref_table"]:
                fq = (
                    entry["ref_table"]
                    if entry["ref_schema"] == "public"
                    else f"{entry['ref_schema']}.{entry['ref_table']}"
                )
                rec.foreign_keys.append(
                    ForeignKey(cname, entry["columns"], fq, entry["ref_columns"])
                )

        for key, rec in tables.items():
            for other_key, other in tables.items():
                if other_key == key:
                    continue
                for fk in other.foreign_keys:
                    target = fk.ref_table.split(".")[-1] if "." in fk.ref_table else fk.ref_table
                    if other.schema == rec.schema and target == rec.name:
                        src = (
                            other.name if other.schema == "public" else f"{other.schema}.{other.name}"
                        )
                        rec.referenced_by.append(
                            ForeignKey(
                                fk.name,
                                fk.columns,
                                fk.ref_table,
                                fk.ref_columns,
                                source=src,
                            )
                        )

        cur.execute(INDEXES_SQL)
        for row in cur.fetchall():
            tables[_key(row["schemaname"], row["tablename"])].indexes.append(
                (row["indexname"], row["indexdef"])
            )

        cur.execute(VIEWDEFS_SQL)
        for row in cur.fetchall():
            rec = tables.get(_key(row["schema"], row["table_name"]))
            if rec:
                rec.view_def = row["view_def"]

    for rec in tables.values():
        pk_cols = set(rec.primary_key)
        fk_refs: dict[str, str] = {}
        for fk in rec.foreign_keys:
            for col_name, ref_col in zip(fk.columns, fk.ref_columns):
                fk_refs[col_name] = f"{fk.ref_table}({ref_col})"
        for col in rec.columns:
            col.pk = col.name in pk_cols
            col.fk_ref = fk_refs.get(col.name)

    return tables
