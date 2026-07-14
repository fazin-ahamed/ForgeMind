from pathlib import Path
from types import SimpleNamespace

import pytest

from forgemind.domain import SourceRecord
from forgemind.ingest import chunk_source, discover_text_sources, ingest_project, parse_git_log
from forgemind.store import ForgeStore


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def test_discovery_skips_binary_secret_vendor_and_escaping_symlink(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-12345678901234567890\n", encoding="utf-8"
    )
    (tmp_path / "image.bin").write_bytes(b"abc\x00def")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("vendor", encoding="utf-8")
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("must not ingest", encoding="utf-8")
    try:
        (tmp_path / "escape.txt").symlink_to(outside)
    except OSError:
        pass

    sources = discover_text_sources(tmp_path)

    assert [source.path for source in sources] == ["src/app.py"]


def test_discovery_only_applies_exclusions_below_selected_root(tmp_path: Path) -> None:
    root = tmp_path / ".worktrees" / "project"
    root.mkdir(parents=True)
    (root / "app.py").write_text("print('safe')\n", encoding="utf-8")

    assert [source.path for source in discover_text_sources(root)] == ["app.py"]


def test_python_functions_become_line_addressable_chunks() -> None:
    source = SourceRecord.from_text(
        "auth.py",
        "def login(user):\n    return user.id\n\ndef logout(user):\n    return None\n",
        1,
    )

    chunks = chunk_source(source)

    assert [(chunk.symbol, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        ("login", 1, 2),
        ("logout", 4, 5),
    ]


def test_chunking_keeps_parse_tree_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class State:
        alive = True

    class Node:
        def __init__(
            self,
            state: State,
            node_type: str,
            children: list["Node"] | None = None,
        ) -> None:
            self.state = state
            self.type = node_type
            self._children = children or []
            self.start_point = SimpleNamespace(row=0)
            self.end_point = SimpleNamespace(row=1)
            self.start_byte = 4
            self.end_byte = 7

        @property
        def children(self) -> list["Node"]:
            if not self.state.alive:
                raise RuntimeError("parse tree released")
            return self._children

        def child_by_field_name(self, _field: str) -> "Node":
            if not self.state.alive:
                raise RuntimeError("parse tree released")
            return Node(self.state, "identifier")

    class Tree:
        def __init__(self) -> None:
            self.state = State()
            function = Node(self.state, "function_definition")
            function.start_byte = 0
            function.end_byte = len("def run():\n    return 1")
            self.root_node = Node(self.state, "module", [function])

        def __del__(self) -> None:
            self.state.alive = False

    class Parser:
        def parse(self, _source: bytes) -> Tree:
            return Tree()

    monkeypatch.setattr("forgemind.ingest.get_parser", lambda _language: Parser())
    source = SourceRecord.from_text("app.py", "def run():\n    return 1\n", 1)

    chunks = chunk_source(source)

    assert [(chunk.symbol, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        ("run", 1, 2)
    ]


def test_chunking_splits_source_lines_once() -> None:
    class CountingText(str):
        calls = 0

        def splitlines(self, keepends: bool = False) -> list[str]:
            self.calls += 1
            return super().splitlines(keepends)

    text = CountingText(
        "def first():\n    return 1\n\ndef second():\n    return 2\n"
    )
    source = SourceRecord("source", "app.py", "sha", 1, text)

    chunks = chunk_source(source)

    assert len(chunks) == 2
    assert text.calls == 1


def test_parse_git_log_creates_temporal_events() -> None:
    events = parse_git_log(
        "abc123\x1f2026-04-18T10:00:00+00:00\x1fMigrate users to UUID\x00"
    )
    assert events[0].commit == "abc123"
    assert events[0].occurred_at == "2026-04-18T10:00:00+00:00"


def test_parse_git_log_strips_record_separator_newlines() -> None:
    events = parse_git_log(
        "a\x1f2026-04-18T10:00:00+00:00\x1ffirst\x00\n"
        "b\x1f2026-04-19T10:00:00+00:00\x1fsecond\x00\n"
    )
    assert [event.commit for event in events] == ["a", "b"]


def test_ingest_project_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)

    first = ingest_project(root, store, FakeEmbedder())
    second = ingest_project(root, store, FakeEmbedder())

    assert first == {"sources": 1, "chunks": 1, "events": 0}
    assert second == {"sources": 1, "chunks": 0, "events": 0}
    assert store.count("sources") == 1
    assert store.count("chunks") == 1


def test_ingestion_rolls_back_every_write_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source_path = root / "app.py"
    source_path.write_text("def run():\n    return 1\n", encoding="utf-8")
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    ingest_project(root, store, FakeEmbedder())
    source_path.write_text("def run():\n    return 2\n", encoding="utf-8")

    monkeypatch.setattr(store, "put_embedding", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        ingest_project(root, store, FakeEmbedder())

    assert store.count("sources") == 1
    assert store.count("chunks") == 1
