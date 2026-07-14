from __future__ import annotations

from collections import defaultdict

from sentence_transformers import SentenceTransformer

from forgemind.domain import SearchHit
from forgemind.store import ForgeStore


EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDER_REVISION = "5c38ec7"


class Embedder:
    def __init__(
        self,
        model_name: str = EMBEDDER_MODEL,
        revision: str = EMBEDDER_REVISION,
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            revision=revision,
        )
        dimensions = self.model.get_embedding_dimension()
        if dimensions is None:
            raise ValueError("embedding model does not declare its dimensions")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.astype("float32").tolist() for vector in vectors]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class Retriever:
    def __init__(self, store: ForgeStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        lexical = self.store.fts_search(query, limit * 2)
        vector = self.embedder.encode([query])[0]
        semantic = [
            chunk_id
            for chunk_id, _distance in self.store.vector_search(vector, limit * 2)
        ]
        fused = rrf([lexical, semantic])[:limit]
        return self._hits(fused, (("lexical", lexical), ("semantic", semantic)))

    def search_vector(self, query: str, limit: int = 20) -> list[SearchHit]:
        vector = self.embedder.encode([query])[0]
        ranked = self.store.vector_search(vector, limit)
        ids = [chunk_id for chunk_id, _distance in ranked]
        scored = [(chunk_id, 1.0 / (1.0 + distance)) for chunk_id, distance in ranked]
        return self._hits(scored, (("semantic", ids),))

    def _hits(
        self,
        ranked: list[tuple[str, float]],
        rankings: tuple[tuple[str, list[str]], ...],
    ) -> list[SearchHit]:
        rows = self.store.chunks_by_ids([chunk_id for chunk_id, _score in ranked])
        hits: list[SearchHit] = []
        for chunk_id, score in ranked:
            row = rows.get(chunk_id)
            if row is None:
                continue
            channels = tuple(
                name
                for name, ranking in rankings
                if chunk_id in ranking
            )
            hits.append(
                SearchHit(
                    chunk_id,
                    row["source_id"],
                    row["source_sha256"],
                    row["path"],
                    row["start_line"],
                    row["end_line"],
                    row["text"],
                    score,
                    channels,
                )
            )
        return hits
