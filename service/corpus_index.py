"""
CorpusIndex: builds a flat in-memory embedding matrix over all corpus sentences
and answers top-k semantic similarity queries via cosine similarity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from service.embeddings import BaseEmbedder
from service.schemas import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    sentence: str
    doc: str
    is_unfair: bool


class CorpusIndex:
    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
        self._entries: list[_Entry] = []
        self._matrix = None  # sparse CSR (tfidf) or dense ndarray (sbert), shape (N, dim), L2-normalised

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, docs: list[dict]) -> None:
        """
        Fit embedder on all corpus sentences, build embedding matrix.

        docs: list of dicts from load_corpus(), each with:
            sentences: list[str]
            name: str
            labels: list[int]   (1=unfair, -1=fair)
        """
        logger.info("Building corpus index (%s)...", self._embedder.name)

        entries: list[_Entry] = []
        all_sentences: list[str] = []

        for doc in docs:
            for sent, lbl in zip(doc['sentences'], doc['labels']):
                entries.append(_Entry(
                    sentence=sent,
                    doc=doc['name'],
                    is_unfair=(lbl == 1),
                ))
                all_sentences.append(sent)

        self._entries = entries

        # Fit then encode (TF-IDF needs fit; SBERT ignores fit but pre-warms)
        self._embedder.fit(all_sentences)
        self._matrix = self._embedder.encode(all_sentences)  # already L2-normalised

        logger.info("Corpus index ready: %d sentences, dim=%d",
                    len(entries), self._matrix.shape[1])

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top_k most similar sentences to query (cosine similarity)."""
        if self._matrix is None:
            raise RuntimeError("CorpusIndex not built yet. Call build() first.")

        top_k = min(top_k, len(self._entries))

        # Encode query (already L2-normalised by embedder)
        q_vec = self._embedder.encode_one(query)  # shape (dim,)

        # Cosine similarity = dot product (both sides L2-normalised)
        scores = np.asarray(self._matrix @ q_vec).ravel()  # shape (N,)

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            entry = self._entries[idx]
            results.append(SearchResult(
                sentence=entry.sentence,
                doc=entry.doc,
                score=float(scores[idx]),
                is_unfair=entry.is_unfair,
            ))
        return results

    @property
    def size(self) -> int:
        return len(self._entries)
