"""Real in-memory stores for anti-mocking tests.

Uses real SQLite in-memory databases instead of mocking store interfaces.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional


class InMemoryStore:
    """Simple key-value store backed by SQLite in-memory database.

    Provides the same interface pattern as OptimizationStore and
    TelemetryStore but runs entirely in-memory for fast tests.
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  created_at REAL DEFAULT (julianday('now'))"
            ")"
        )
        self._conn.commit()

    def put(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        self._conn.commit()

    def keys(self) -> List[str]:
        rows = self._conn.execute("SELECT key FROM kv ORDER BY key").fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()


def make_temp_db(tmp_path: Path, name: str = "test.db") -> Path:
    """Create a temporary SQLite database path for tests that need file-based stores."""
    db_path = tmp_path / name
    return db_path
