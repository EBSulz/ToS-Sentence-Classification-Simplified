from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize

from src.preprocessing import clean
from src.features import make_vectorizer


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, sentences: list[str]) -> None: ...

    @abstractmethod
    def encode(self, sentences: list[str]) -> np.ndarray: ...

    def encode_one(self, sentence: str) -> np.ndarray:
        return self.encode([sentence])[0]


class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF bag-of-words embedder with L2 normalisation."""

    name = "tfidf"

    def __init__(self) -> None:
        self._vec = make_vectorizer()
        self._fitted = False

    def fit(self, sentences: list[str]) -> None:
        cleaned = [clean(s) for s in sentences]
        self._vec.fit(cleaned)
        self._fitted = True

    def encode(self, sentences: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder must be fit before encode()")
        cleaned = [clean(s) for s in sentences]
        sparse = self._vec.transform(cleaned)
        dense = sparse.toarray().astype(np.float32)
        return normalize(dense, norm="l2")


class SBERTEmbedder(BaseEmbedder):
    """sentence-transformers all-MiniLM-L6-v2 embedder with L2 normalisation."""

    name = "sbert"

    def __init__(self) -> None:
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, sentences: list[str]) -> None:
        self._load()  # pre-warm the model

    def encode(self, sentences: list[str]) -> np.ndarray:
        self._load()
        vecs = self._model.encode(sentences, show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


def get_embedder(method: str) -> BaseEmbedder:
    """Factory: 'tfidf' or 'sbert'."""
    if method == "tfidf":
        return TFIDFEmbedder()
    if method == "sbert":
        return SBERTEmbedder()
    raise ValueError(f"Unknown embedding method: {method!r}. Choose 'tfidf' or 'sbert'.")
