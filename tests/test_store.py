from pathlib import Path

from forgemind.domain import ChunkRecord, ProjectEvent, SourceRecord
from forgemind.store import ForgeStore


def test_source_round_trip_is_idempotent(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("src/auth.py", "def login():\n    return True\n", 123)

    store.upsert_source(source)
    store.upsert_source(source)

    assert store.source(source.id) == source
    assert store.count("sources") == 1


def test_chunks_replace_atomically_and_events_are_idempotent(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("auth.py", "one\ntwo\n", 1)
    store.upsert_source(source)
    first = ChunkRecord("c1", source.id, source.path, 1, 1, "one")
    second = ChunkRecord("c2", source.id, source.path, 2, 2, "two")

    store.replace_chunks(source.id, [first])
    store.replace_chunks(source.id, [second])
    event = ProjectEvent("e1", "abc", "2026-04-18T00:00:00Z", "migration")
    store.upsert_events([event, event])

    assert store.count("chunks") == 1
    assert store.count("events") == 1


def test_active_hits_are_deterministically_path_ordered(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    second = SourceRecord.from_text("z.py", "second", 1)
    first = SourceRecord.from_text("a.py", "first", 1)
    for source, chunk_id in ((second, "c2"), (first, "c1")):
        store.upsert_source(source)
        store.replace_chunks(
            source.id,
            [ChunkRecord(chunk_id, source.id, source.path, 1, 1, source.text)],
        )

    assert [hit.path for hit in store.active_hits()] == ["a.py", "z.py"]
