from __future__ import annotations

import re


class SQLGuardError(ValueError):
    pass


FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|truncate|drop|alter|create|grant|revoke|copy|vacuum|analyze|analyse"
    r"|call|do|set|reset|listen|notify|load|reindex|cluster|comment|lock|merge|execute|prepare"
    r"|deallocate|declare|checkpoint|discard|import|refresh|abort|begin|commit"
    r"|rollback|savepoint|release)\b",
    re.IGNORECASE,
)

DANGEROUS_FUNCTIONS = (
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_current_logfile",
    "lo_import",
    "lo_export",
    "lo_creat",
    "lo_create",
    "lo_unlink",
    "dblink",
    "pg_advisory_lock",
    "pg_advisory_xact_lock",
    "pg_advisory_unlock",
    "pg_backup_start",
    "pg_backup_stop",
    "pg_switch_wal",
    "pg_create_restore_point",
)

LINE_COMMENT = re.compile(r"--[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
QUOTED_STRING = re.compile(r"'(?:[^']|'')*'|\$(?!\w)(?:[^$]|\$(?!\$))*\$\$")
IDENTIFIER_QUOTED = re.compile(r'"[^"]*"')


def _strip_noise(sql: str) -> tuple[str, str]:
    without_comments = BLOCK_COMMENT.sub(" ", LINE_COMMENT.sub(" ", sql))
    literals_stripped = QUOTED_STRING.sub("''", without_comments)
    return without_comments, literals_stripped


def validate_sql(sql: str) -> str:
    cleaned = (sql or "").strip()
    if not cleaned:
        raise SQLGuardError("Empty SQL statement.")
    cleaned = cleaned.rstrip(";").rstrip()

    without_comments, literal_free = _strip_noise(cleaned)
    literal_free = IDENTIFIER_QUOTED.sub('""', literal_free)
    without_comments = without_comments.strip()

    if ";" in literal_free:
        raise SQLGuardError("Only a single statement is allowed.")

    first_word = re.match(r"^(\w+)", without_comments, re.IGNORECASE)
    if not first_word or first_word.group(1).upper() not in ("SELECT", "WITH"):
        raise SQLGuardError("Statement must start with SELECT or WITH.")

    match = FORBIDDEN_KEYWORDS.search(literal_free)
    if match:
        raise SQLGuardError(f"Forbidden keyword in read-only context: {match.group(1).upper()}")

    for func in DANGEROUS_FUNCTIONS:
        if func in literal_free.lower():
            raise SQLGuardError(f"Forbidden function call: {func}()")

    return without_comments
