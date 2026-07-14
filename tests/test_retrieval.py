from pathlib import Path

import forgemind.retrieval as retrieval
from forgemind.domain import ChunkRecord, SourceRecord
from forgemind.retrieval import EMBEDDER_REVISION, Embedder, Retriever, rrf
from forgemind.store import ForgeStore


def test_embedder_loads_pinned_model_revision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model_name, device, revision) -> None:
            captured.update(
                model_name=model_name,
                device=device,
                revision=revision,
            )

        def get_embedding_dimension(self) -> int:
            return 3

    monkeypatch.setattr(retrieval, "SentenceTransformer", FakeModel)

    Embedder()

    assert captured["revision"] == EMBEDDER_REVISION
    assert len(EMBEDDER_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in EMBEDDER_REVISION)


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


def test_vector_only_retrieval_does_not_use_lexical_channel(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    source = SourceRecord.from_text("session.py", "UUID migration", 1)
    chunk = ChunkRecord("chunk-1", source.id, source.path, 1, 1, source.text)
    store.upsert_source(source)
    store.replace_chunks(source.id, [chunk])
    store.put_embedding(chunk.id, [1.0, 0.0, 0.0])

    class FixedEmbedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    hits = Retriever(store, FixedEmbedder()).search_vector("uuid", 1)

    assert [hit.chunk_id for hit in hits] == ["chunk-1"]
    assert hits[0].channels == ("semantic",)
