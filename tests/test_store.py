from pathlib import Path

from forgemind.domain import SourceRecord
from forgemind.store import ForgeStore


def test_source_round_trip_is_idempotent(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("src/auth.py", "def login():\n    return True\n", 123)

    store.upsert_source(source)
    store.upsert_source(source)

    assert store.source(source.id) == source
    assert store.count("sources") == 1
