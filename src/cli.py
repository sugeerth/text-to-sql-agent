"""Interactive text-to-SQL CLI. Run with: uv run cli (or python -m src.cli)"""

import argparse
import sys

from dotenv import load_dotenv

from src.agent import AgentConfig, SqlAgent
from src.db import DEFAULT_DB, connect, print_schema

MAX_DISPLAY_ROWS = 20

HELP = """\
Ask a question about the database in plain English.
Commands:
  /schema [table]   show the database schema (or one table)
  /reset            clear conversation context
  /help             show this help
  exit | quit       leave
"""


def render(result) -> None:
    if result.sql:
        print(f"\nSQL:\n```sql\n{result.sql}\n```")
    if not result.ok:
        print(f"\nSorry, I couldn't answer that ({result.attempts} attempt(s)).")
        print(f"Last error: {result.error}")
        return
    df = result.df
    print()
    if len(df) == 0:
        print("(0 rows)")
    else:
        print(df.head(MAX_DISPLAY_ROWS).to_string(index=False))
        if len(df) > MAX_DISPLAY_ROWS:
            print(f"... {len(df) - MAX_DISPLAY_ROWS} more rows")
    print(f"\n[{len(df)} row(s) in {result.latency_s:.1f}s]")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Natural-language SQL assistant")
    parser.add_argument("--model", help="model id override (default: LLM_MODEL)")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    args = parser.parse_args()

    config = AgentConfig(**({"model": args.model} if args.model else {}))
    try:
        agent = SqlAgent(db_path=args.db, config=config)
    except FileNotFoundError as e:
        sys.exit(str(e))
    schema_conn = connect(args.db, read_only=True)  # for /schema only

    print("=" * 60)
    print("Text-to-SQL CLI  |  Chinook music store")
    print(f"Model: {config.model.split('/')[-1] or '(LLM_MODEL not set)'}")
    print("Type a question, /help for commands, exit to quit.")
    print("=" * 60)

    while True:
        try:
            line = input("\nask> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not line:
            continue
        lowered = line.lower()
        if lowered in ("exit", "quit"):
            print("Bye!")
            break
        if lowered == "/help":
            print(HELP)
        elif lowered == "/reset":
            agent.reset()
            print("Context cleared.")
        elif lowered.startswith("/schema"):
            parts = line.split(maxsplit=1)
            print_schema(schema_conn, parts[1] if len(parts) > 1 else None)
        else:
            render(agent.ask(line))


if __name__ == "__main__":
    main()
