"""SQLite helpers: connections, queries, and schema introspection."""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DB = "data/Chinook.db"


def connect(db_path: str = DEFAULT_DB, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite database. Fails clearly instead of silently creating an empty file."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run `uv run setup-db` to download the Chinook sample database."
        )
    if read_only:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return sqlite3.connect(db_path)


def query(conn: sqlite3.Connection, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    return pd.read_sql_query(sql, conn, params=params)


def rows(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Run a query and return a list of {column: value} dicts."""
    cursor = conn.execute(sql)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_schema(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """{table: [PRAGMA table_info rows]} for every table, sorted by name."""
    tables = rows(conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return {t["name"]: rows(conn, f"PRAGMA table_info({t['name']})") for t in tables}


def print_schema(conn: sqlite3.Connection, table: Optional[str] = None) -> None:
    """Print all tables (or one) with column name, type, nullability and primary key."""
    schema = get_schema(conn)
    if table and table not in schema:
        print(f"No table named '{table}'. Tables: {', '.join(schema)}")
        return
    shown = {table: schema[table]} if table else schema
    if not table:
        print(f"\n{len(schema)} tables: {', '.join(schema)}")
    for name, columns in shown.items():
        print(f"\n{name}")
        print(f"  {'column':<24} {'type':<16} {'null':<9} pk")
        for col in columns:
            null = "not null" if col["notnull"] else ""
            pk = "yes" if col["pk"] else ""
            print(f"  {col['name']:<24} {col['type']:<16} {null:<9} {pk}")
    print()
