import pytest

from rag.config import load_settings
from rag.dialects import canonical_name, get_dialect
from rag.dialects.mysql import build_mysql_catalog
from rag.dialects.sqlserver import build_mssql_catalog
from rag.pipeline import render_answer_system, render_sql_system


def _settings(**overrides):
    base = load_settings()
    return base.__class__(**{**base.__dict__, **overrides})


def test_canonical_aliases():
    assert canonical_name("postgresql") == "postgres"
    assert canonical_name("mariadb") == "mysql"
    assert canonical_name("mssql") == "sqlserver"
    assert canonical_name("bq") == "bigquery"
    assert canonical_name(None) == "postgres"


def test_unknown_dialect_raises():
    with pytest.raises(ValueError):
        canonical_name("excel")


def test_get_dialect_returns_right_class():
    s = _settings(db_dialect="postgres")
    d = get_dialect(s)
    assert d.label == "PostgreSQL"
    assert d.supports_odoo_metadata

    s = _settings(db_dialect="mysql")
    assert get_dialect(s).label == "MySQL/MariaDB"

    s = _settings(db_dialect="snowflake")
    assert get_dialect(s).label == "Snowflake"


def test_mysql_catalog_builder():
    tables = [
        {"ts": "shop", "tn": "customers", "tt": "BASE TABLE", "est_rows": 10, "tc": "Buyers"},
        {"ts": "shop", "tn": "orders", "tt": "BASE TABLE", "est_rows": 100, "tc": None},
        {"ts": "shop", "tn": "v_big", "tt": "VIEW", "est_rows": 0, "tc": None},
    ]
    columns = [
        {"ts": "shop", "tn": "customers", "cn": "id", "ct": "int", "nul": "NO", "cd": None, "cc": "", "ex": ""},
        {"ts": "shop", "tn": "customers", "cn": "name", "ct": "varchar(80)", "nul": "YES", "cd": None, "cc": "Full name", "ex": ""},
        {"ts": "shop", "tn": "orders", "cn": "id", "ct": "int", "nul": "NO", "cd": None, "cc": None, "ex": "auto_increment"},
        {"ts": "shop", "tn": "orders", "cn": "customer_id", "ct": "int", "nul": "YES", "cd": None, "cc": None, "ex": ""},
        {"ts": "shop", "tn": "v_big", "cn": "id", "ct": "int", "nul": "YES", "cd": None, "cc": None, "ex": ""},
    ]
    keys = [
        {"ts": "shop", "tn": "customers", "cn": "PRIMARY", "ctype": "PRIMARY KEY", "col": "id"},
        {"ts": "shop", "tn": "orders", "cn": "PRIMARY", "ctype": "PRIMARY KEY", "col": "id"},
        {"ts": "shop", "tn": "orders", "cn": "uq_ref", "ctype": "UNIQUE", "col": "ref"},
    ]
    fks = [
        {"ts": "shop", "tn": "orders", "col": "customer_id", "cn": "fk_cust",
         "rts": "shop", "rtn": "customers", "rcol": "id"},
    ]
    indexes = [
        {"ts": "shop", "tn": "orders", "ix": "ix_cust", "cols": "customer_id", "nu": 1},
    ]
    viewdefs = [{"ts": "shop", "tn": "v_big", "vdef": "SELECT 1"}]

    cat = build_mysql_catalog(tables, columns, keys, fks, indexes, viewdefs)
    customers = cat[("shop", "customers")]
    orders = cat[("shop", "orders")]

    assert customers.comment == "Buyers"
    assert customers.primary_key == ["id"]
    assert customers.columns[1].comment == "Full name"
    assert orders.columns[0].identity is True
    assert orders.primary_key == ["id"]
    assert orders.unique_constraints == [["ref"]]
    assert orders.foreign_keys[0].ref_table == "customers"
    assert customers.referenced_by[0].source == "orders"
    assert orders.indexes[0][0] == "ix_cust"
    assert cat[("shop", "v_big")].kind_label == "VIEW"
    assert cat[("shop", "v_big")].view_def == "SELECT 1"
    assert orders.columns[1].fk_ref == "customers(id)"


def test_mssql_catalog_builder():
    tables = [
        {"ts": "dbo", "tn": "invoices", "est_rows": 42, "tc": "Billing docs"},
        {"ts": "sales", "tn": "customers", "est_rows": 7, "tc": None},
    ]
    views = [{"ts": "dbo", "tn": "vw_all", "vdef": "AS SELECT 1"}]
    columns = [
        {"ts": "dbo", "tn": "invoices", "cn": "id", "dtype": "int", "maxlen": None, "chars": None,
         "prec": 10, "sc": 0, "nul": False, "cdef": None, "ident": True, "comp": False, "cc": None},
        {"ts": "dbo", "tn": "invoices", "cn": "total", "dtype": "decimal", "maxlen": None, "chars": None,
         "prec": 18, "sc": 2, "nul": True, "cdef": "((0))", "ident": False, "comp": False, "cc": None},
        {"ts": "dbo", "tn": "invoices", "cn": "cust_id", "dtype": "nvarchar", "maxlen": 60, "chars": 30,
         "prec": 0, "sc": 0, "nul": True, "cdef": None, "ident": False, "comp": False, "cc": None},
        {"ts": "sales", "tn": "customers", "cn": "id", "dtype": "int", "maxlen": None, "chars": None,
         "prec": 10, "sc": 0, "nul": False, "cdef": None, "ident": False, "comp": False, "cc": None},
    ]
    keys = [{"ts": "dbo", "tn": "invoices", "cn": "PK_invoices", "ctype": "PRIMARY KEY", "col": "id"}]
    fks = [
        {"ts": "dbo", "tn": "invoices", "col": "cust_id", "cn": "FK_inv_cust",
         "rts": "sales", "rtn": "customers", "rcol": "id"},
    ]
    indexes = [{"ts": "dbo", "tn": "invoices", "ix": "IX_total", "uniq": False, "cols": "total"}]

    cat = build_mssql_catalog(tables, views, columns, keys, fks, indexes)
    inv = cat[("dbo", "invoices")]
    cust = cat[("sales", "customers")]

    assert inv.row_estimate == 42
    assert inv.comment == "Billing docs"
    assert inv.columns[0].type == "int"
    assert inv.columns[0].identity is True
    assert inv.columns[1].type == "decimal(18,2)"
    assert inv.columns[2].type == "nvarchar(30)"
    assert inv.foreign_keys[0].ref_table == "sales.customers"
    assert cust.referenced_by[0].source == "dbo.invoices"
    assert cat[("dbo", "vw_all")].kind_label == "VIEW"


def test_prompt_rendering_substitutes_tokens():
    text = render_sql_system("MySQL/MariaDB", "- backticks hint", 50)
    assert "__ENGINE__" not in text
    assert "__MAX_ROWS__" not in text
    assert "__HINTS__" not in text
    assert "senior MySQL/MariaDB analyst" in text
    assert "<= 50" in text
    assert "backticks hint" in text
    assert render_answer_system("Snowflake").startswith("You answer questions about a Snowflake database")


def test_generic_dialect_url_compose():
    from rag.dialects.generic import GenericSQLAlchemyDialect

    s = _settings(
        db_dialect="duckdb", db_user="", db_password="", db_host="", db_port=0, db_name="/tmp/opencode/x.duckdb"
    )
    d = GenericSQLAlchemyDialect(s, scheme_key="duckdb")
    assert d._url() == "duckdb:////tmp/opencode/x.duckdb"
