from pathlib import Path

from forgemind.ingest import ingest_project
from forgemind.store import ForgeStore


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_unchanged_source_is_not_reembedded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "app.py"
    path.write_text("value = 1\n", encoding="utf-8")
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    embedder = CountingEmbedder()

    ingest_project(root, store, embedder)
    first_calls = embedder.calls
    second = ingest_project(root, store, embedder)

    assert embedder.calls == first_calls
    assert second["chunks"] == 0


def test_changed_source_replaces_active_chunks_but_keeps_revision(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "app.py"
    path.write_text("value = 1\n", encoding="utf-8")
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    embedder = CountingEmbedder()

    ingest_project(root, store, embedder)
    previous = store.current_source("app.py")
    path.write_text("value = 2\n", encoding="utf-8")
    changed = ingest_project(root, store, embedder)

    current = store.current_source("app.py")
    assert previous is not None and current is not None
    assert previous.id != current.id
    assert store.source(previous.id) == previous
    assert store.count("sources") == 2
    assert store.count("chunks") == 1
    assert changed["chunks"] == 1


def test_deleted_source_is_retired_from_active_index_but_stays_archived(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "app.py"
    path.write_text("value = 1\n", encoding="utf-8")
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    embedder = CountingEmbedder()

    ingest_project(root, store, embedder)
    archived = store.current_source("app.py")
    path.unlink()
    ingest_project(root, store, embedder)

    assert archived is not None
    assert store.current_source("app.py") is None
    assert store.source(archived.id) == archived
    assert store.count("sources") == 1
    assert store.count("chunks") == 0
