import pytest

from rag.sqlguard import SQLGuardError, validate_sql


def test_simple_select_ok():
    sql = "SELECT id, name FROM public.res_partner LIMIT 10;"
    assert validate_sql(sql).startswith("SELECT")


def test_with_cte_ok():
    sql = """
    WITH t AS (SELECT id FROM res_company)
    SELECT * FROM t
    """
    assert "WITH" in validate_sql(sql)


def test_update_rejected():
    with pytest.raises(SQLGuardError):
        validate_sql("UPDATE res_partner SET name = 'x'")


def test_delete_rejected():
    with pytest.raises(SQLGuardError):
        validate_sql("DELETE FROM res_partner")


def test_ddl_rejected():
    for bad in [
        "CREATE TABLE x (id int)",
        "DROP TABLE res_partner",
        "TRUNCATE res_partner",
        "ALTER TABLE res_partner ADD COLUMN x int",
        "GRANT ALL ON res_partner TO public",
        "VACUUM ANALYZE",
    ]:
        with pytest.raises(SQLGuardError):
            validate_sql(bad)


def test_literal_containing_keyword_allowed():
    sql = "SELECT id FROM mail_message WHERE state = 'delete'"
    assert validate_sql(sql)


def test_quoted_identifier_comment_column_allowed():
    sql = 'SELECT "comment" FROM my_table'
    assert validate_sql(sql)


def test_multiple_statements_rejected():
    with pytest.raises(SQLGuardError):
        validate_sql("SELECT 1; DROP TABLE x")


def test_semicolon_in_string_ok():
    sql = "SELECT * FROM t WHERE note = 'a;b'"
    assert validate_sql(sql)


def test_dangerous_function_rejected():
    with pytest.raises(SQLGuardError):
        validate_sql("SELECT pg_sleep(10)")


def test_non_select_rejected():
    with pytest.raises(SQLGuardError):
        validate_sql("EXPLAIN ANALYZE SELECT 1")
    with pytest.raises(SQLGuardError):
        validate_sql("DO $$ BEGIN NULL; END $$;")


def test_comments_stripped_before_checks():
    sql = "-- this mentions DELETE\nSELECT 1 /* and UPDATE here */"
    assert validate_sql(sql) == "SELECT 1"
