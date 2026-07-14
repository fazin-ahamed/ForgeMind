from __future__ import annotations

import sqlite3
import re
from pathlib import Path

import sqlite_vec

from forgemind.domain import ChunkRecord, ProjectEvent, SourceRecord


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

    def replace_chunks(self, source_id: str, chunks: list[ChunkRecord]) -> None:
        old_ids = [
            row[0]
            for row in self.connection.execute(
                "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
            )
        ]
        self.connection.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", ((chunk_id,) for chunk_id in old_ids)
        )
        self.connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        for chunk in chunks:
            self.connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    chunk.source_id,
                    chunk.path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.symbol,
                ),
            )
            self.connection.execute(
                "INSERT INTO chunks_fts(chunk_id, path, symbol, text) VALUES (?, ?, ?, ?)",
                (chunk.id, chunk.path, chunk.symbol or "", chunk.text),
            )

    def upsert_events(self, events: list[ProjectEvent]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
            ((event.id, event.commit, event.occurred_at, event.summary) for event in events),
        )

    def enable_vectors(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("vector dimensions must be positive")
        self.connection.enable_load_extension(True)
        try:
            sqlite_vec.load(self.connection)
        finally:
            self.connection.enable_load_extension(False)
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors "
            f"USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{int(dimensions)}])"
        )

    def put_embedding(self, chunk_id: str, vector: list[float]) -> None:
        self.connection.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        self.connection.execute(
            "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vector)),
        )

    def vector_search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        rows = self.connection.execute(
            "SELECT chunk_id, distance FROM chunk_vectors "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(vector), limit),
        ).fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    def fts_search(self, query: str, limit: int) -> list[str]:
        tokens = re.findall(r"[\w./:-]+", query, flags=re.UNICODE)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.connection.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" for _ in chunk_ids)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE id IN ({marks})", chunk_ids
        ).fetchall()
        return {str(row["id"]): row for row in rows}
