"""Sentence-Transformers wrapper producing L2-normalised float32 vectors (CPU)."""
from __future__ import annotations

import numpy as np

import config


class Embedder:
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:  # lazy: importing torch is slow
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)