from __future__ import annotations

import re
from pathlib import Path

from forgemind.domain import SourceRecord


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


def is_likely_secret(path: str, text: str) -> bool:
    if Path(path).name.lower() in {".env", "credentials.json", "secrets.json"}:
        return True
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def discover_text_sources(root: Path) -> list[SourceRecord]:
    resolved_root = root.resolve(strict=True)
    sources: list[SourceRecord] = []
    for candidate in sorted(resolved_root.rglob("*")):
        if not candidate.is_file() or any(part in EXCLUDED_PARTS for part in candidate.parts):
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
