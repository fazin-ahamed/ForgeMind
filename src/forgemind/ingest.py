from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Protocol

from tree_sitter_language_pack import SupportedLanguage, get_parser

from forgemind.domain import ChunkRecord, ProjectEvent, SourceRecord
from forgemind.store import ForgeStore


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".worktrees",
    ".forgemind-private",
    ".superpowers",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "models",
    "artifacts",
    "benchmark-results",
}
SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    ),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
LANGUAGES: dict[str, SupportedLanguage] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}
FUNCTION_TYPES = {"function_definition", "function_declaration", "method_definition"}


class EmbeddingEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


def is_likely_secret(path: str, text: str) -> bool:
    if Path(path).name.lower() in {".env", "credentials.json", "secrets.json"}:
        return True
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def discover_text_sources(root: Path) -> list[SourceRecord]:
    resolved_root = root.resolve(strict=True)
    sources: list[SourceRecord] = []
    for candidate in sorted(resolved_root.rglob("*")):
        relative_parts = candidate.relative_to(resolved_root).parts
        if not candidate.is_file() or any(part in EXCLUDED_PARTS for part in relative_parts):
            continue
        resolved = candidate.resolve(strict=True)
        if resolved_root not in resolved.parents:
            continue
        raw = resolved.read_bytes()
        if b"\x00" in raw or len(raw) > 2_000_000:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        relative = resolved.relative_to(resolved_root).as_posix()
        if is_likely_secret(relative, text):
            continue
        sources.append(SourceRecord.from_text(relative, text, resolved.stat().st_mtime_ns))
    return sources


def _chunk(source: SourceRecord, start: int, end: int, symbol: str | None) -> ChunkRecord:
    text = "\n".join(source.text.splitlines()[start - 1 : end])
    chunk_id = hashlib.sha256(f"{source.id}:{start}:{end}".encode()).hexdigest()
    return ChunkRecord(chunk_id, source.id, source.path, start, end, text, symbol)


def chunk_source(source: SourceRecord, max_lines: int = 120) -> list[ChunkRecord]:
    language = LANGUAGES.get(Path(source.path).suffix.lower())
    if language:
        source_bytes = source.text.encode("utf-8")
        stack = [get_parser(language).parse(source_bytes).root_node]
        chunks: list[ChunkRecord] = []
        while stack:
            node = stack.pop()
            if node.type in FUNCTION_TYPES:
                name = node.child_by_field_name("name")
                symbol = source_bytes[name.start_byte : name.end_byte].decode() if name else None
                chunks.append(
                    _chunk(source, node.start_point.row + 1, node.end_point.row + 1, symbol)
                )
            else:
                stack.extend(reversed(node.children))
        if chunks:
            return chunks
    line_count = len(source.text.splitlines())
    return [
        _chunk(source, start, min(start + max_lines - 1, line_count), None)
        for start in range(1, line_count + 1, max_lines)
    ]


def parse_git_log(text: str) -> list[ProjectEvent]:
    events: list[ProjectEvent] = []
    for raw_record in text.split("\x00"):
        record = raw_record.strip("\r\n")
        if not record:
            continue
        commit, occurred_at, summary = record.split("\x1f", 2)
        event_id = hashlib.sha256(commit.encode()).hexdigest()
        events.append(ProjectEvent(event_id, commit, occurred_at, summary))
    return events


def read_git_events(root: Path) -> list[ProjectEvent]:
    completed = subprocess.run(
        ["git", "-C", str(root), "log", "--format=%H%x1f%aI%x1f%s%x00"],
        capture_output=True,
        check=True,
        text=True,
    )
    return parse_git_log(completed.stdout)


def ingest_project(
    root: Path, store: ForgeStore, embedder: EmbeddingEncoder
) -> dict[str, int]:
    sources = discover_text_sources(root)
    heads = {source.path: source for source in store.current_sources()}
    changed: list[tuple[SourceRecord, SourceRecord | None, list[ChunkRecord]]] = []
    for source in sources:
        current = heads.get(source.path)
        if current is not None and current.sha256 == source.sha256:
            continue
        changed.append((source, current, chunk_source(source)))
    discovered_paths = {source.path for source in sources}
    retired = [source for path, source in heads.items() if path not in discovered_paths]
    chunks = [chunk for _source, _current, items in changed for chunk in items]
    vectors = embedder.encode([chunk.text for chunk in chunks]) if chunks else []
    events = read_git_events(root) if (root / ".git").exists() else []

    with store.transaction():
        for source in retired:
            store.remove_active_chunks(source.id)
            store.remove_source_head(source.path)
        for source, current, items in changed:
            if current is not None:
                store.remove_active_chunks(current.id)
            store.upsert_source(source)
            store.set_source_head(source)
            store.replace_chunks(source.id, items)
        for chunk, vector in zip(chunks, vectors, strict=True):
            store.put_embedding(chunk.id, vector)
        store.upsert_events(events)

    return {"sources": len(sources), "chunks": len(chunks), "events": len(events)}
