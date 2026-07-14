from pathlib import Path

from forgemind.store import ForgeStore


def test_vector_search_returns_nearest_chunk(tmp_path: Path) -> None:
    store = ForgeStore(tmp_path / "forge.sqlite")
    store.enable_vectors(3)
    store.put_embedding("auth", [1.0, 0.0, 0.0])
    store.put_embedding("billing", [0.0, 1.0, 0.0])

    results = store.vector_search([0.9, 0.1, 0.0], 1)

    assert results[0][0] == "auth"
