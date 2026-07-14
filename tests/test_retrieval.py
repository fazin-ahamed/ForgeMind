from pathlib import Path

from forgemind.domain import ChunkRecord, SourceRecord
from forgemind.retrieval import rrf
from forgemind.store import ForgeStore


def test_vector_search_returns_nearest_chunk(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    store.put_embedding("auth", [1.0, 0.0, 0.0])
    store.put_embedding("billing", [0.0, 1.0, 0.0])

    results = store.vector_search([0.9, 0.1, 0.0], 1)

    assert results[0][0] == "auth"


def test_rrf_rewards_items_found_by_multiple_channels() -> None:
    fused = rrf([["a", "b"], ["b", "c"]])

    assert fused[0][0] == "b"


def test_fts_search_finds_exact_identifier(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    source = SourceRecord.from_text("session.py", "parseInt(session.userId)", 1)
    chunk = ChunkRecord("chunk-1", source.id, source.path, 1, 1, source.text, "decode")
    store.upsert_source(source)
    store.replace_chunks(source.id, [chunk])

    assert store.fts_search("parseInt", 5)[0] == "chunk-1"
