from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.astype("float32").tolist() for vector in vectors]
