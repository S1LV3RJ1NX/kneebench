"""SQLite connection and forward-only migrations."""

from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

DB_FILENAME = "kneebench.sqlite3"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def connect_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def _migrations() -> list[tuple[int, str]]:
    folder = files("kneebench.store").joinpath("migrations")
    found = []
    for entry in folder.iterdir():
        if entry.name.endswith(".sql"):
            found.append((int(entry.name.split("_", 1)[0]), entry.read_text()))
    return sorted(found)


def migrate(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    current = max(applied, default=0)
    for version, sql in _migrations():
        if version in applied:
            continue
        conn.execute("BEGIN")
        try:
            _exec_script(conn, sql)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        current = version
    return current


def _exec_script(conn: sqlite3.Connection, sql: str) -> None:
    # executescript() commits implicitly, so statements run one by one inside the transaction.
    # Migrations keep comments on their own lines and never put ";" inside string literals.
    code = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    for statement in code.split(";"):
        if statement.strip():
            conn.execute(statement)
