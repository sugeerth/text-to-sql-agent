"""Download the Chinook sample database into data/Chinook.db.

Chinook is a public sample database of a digital media store (artists, albums,
tracks, customers, invoices): https://github.com/lerocha/chinook-database
Run: uv run setup-db
"""

import sqlite3
import sys
import urllib.request
from pathlib import Path

SOURCE = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)
TARGET = Path("data/Chinook.db")


def main() -> None:
    if TARGET.exists() and "--force" not in sys.argv:
        print(f"{TARGET} already exists (use --force to rebuild).")
        return
    print(f"Downloading Chinook from {SOURCE} ...")
    with urllib.request.urlopen(SOURCE, timeout=60) as resp:
        script = resp.read().decode("utf-8-sig")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".db.partial")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    conn.executescript(script)
    conn.commit()
    conn.close()
    tmp.replace(TARGET)
    print(f"Created {TARGET}")


if __name__ == "__main__":
    main()
