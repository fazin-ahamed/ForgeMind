from __future__ import annotations

import sqlite3
from pathlib import Path

from forgemind.domain import SourceRecord


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    modified_ns INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(path, sha256)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    symbol TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    path,
    symbol,
    text,
    tokenize='unicode61 tokenchars ''_./:-'''
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    commit_hash TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    source_id TEXT NOT NULL REFERENCES sources(id),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    PRIMARY KEY(source_id, subject, predicate, object, start_line)
);
"""


class ForgeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def upsert_source(self, source: SourceRecord) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO sources VALUES (?, ?, ?, ?, ?)",
            (source.id, source.path, source.sha256, source.modified_ns, source.text),
        )

    def source(self, source_id: str) -> SourceRecord | None:
        row = self.connection.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return SourceRecord(**dict(row)) if row else None

    def count(self, table: str) -> int:
        if table not in {"sources", "chunks", "events", "relations"}:
            raise ValueError(f"unsupported table: {table}")
        return int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
