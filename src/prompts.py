"""Prompt templates and schema-context construction.

Kept separate from agent logic so prompt tuning never touches control flow.
"""

import sqlite3

from src.db import get_schema, rows

SYSTEM_PROMPT = """You are an expert SQLite analyst. Convert the user's question into a single SQLite query.

DATABASE SCHEMA
{schema_context}

RULES
- Output exactly one SQLite SELECT (or WITH ... SELECT) statement inside a ```sql fence. No prose, no explanation.
- Use only tables and columns that appear in the schema, spelled exactly as shown.
- Dates are stored as TEXT like 'YYYY-MM-DD HH:MM:SS'; filter by year with strftime('%Y', column) = 'YYYY'.
- When aggregating per entity, GROUP BY its primary key (names are not guaranteed unique), but never SELECT id/primary-key columns — output only human-readable columns. Example: for "each playlist" GROUP BY p.PlaylistId but SELECT only p.Name, COUNT(...).
- SELECT exactly the columns the question asks for — no extra columns, no ids.
- When listing records, keep columns as they exist in the schema (FirstName and LastName stay separate). Only when aggregating or ranking per person, combine into one identity column: FirstName || ' ' || LastName.
- Use LEFT JOIN when the question implies entities with zero matches should still appear (e.g. "for each X").
- For rankings, use RANK() OVER (...) so ties share a rank.
- Add LIMIT only when the user asks for a top/first N.
- For follow-up questions, modify the previous SQL to satisfy the refinement.
"""


def build_schema_context(conn: sqlite3.Connection, sample_rows: int = 3) -> str:
    """Render a compact schema description: columns, types, PK/FK annotations,
    plus a few sample rows per table (sample rows reveal storage formats the
    type column can't, e.g. that dates are TEXT)."""
    schema = get_schema(conn)
    lines = []
    for table, columns in schema.items():
        # Foreign keys for this table: {from_column: "OtherTable.OtherColumn"}
        fks = {
            fk["from"]: f'{fk["table"]}.{fk["to"]}'
            for fk in rows(conn, f"PRAGMA foreign_key_list({table})")
        }
        col_parts = []
        for col in columns:
            part = f'{col["name"]} {col["type"]}'
            if col["pk"]:
                part += " PK"
            if col["name"] in fks:
                part += f' -> {fks[col["name"]]}'
            col_parts.append(part)
        lines.append(f'{table}({", ".join(col_parts)})')

        if sample_rows:
            for row in rows(conn, f"SELECT * FROM {table} LIMIT {sample_rows}"):
                values = ", ".join(repr(v) for v in row.values())
                lines.append(f"  sample: ({values})")
    return "\n".join(lines)


def build_system_prompt(conn: sqlite3.Connection) -> str:
    return SYSTEM_PROMPT.format(schema_context=build_schema_context(conn))
