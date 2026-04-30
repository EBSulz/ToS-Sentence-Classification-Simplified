"""FastAPI application for ToS semantic similarity and unfairness classification."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.data_loader import load_corpus
from service.embeddings import get_embedder
from service.corpus_index import CorpusIndex
from service.classifier import C2Classifier
from service.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    ClassifyRequest, ClassifyResponse,
    HealthResponse,
    SearchRequest, SearchResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from env vars ──────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "ToS"))
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", "data/models"))
EMBEDDING_METHOD = os.getenv("EMBEDDING_METHOD", "tfidf")

# ── Singletons (populated in lifespan) ───────────────────────────────────────

_index: CorpusIndex | None = None
_classifier: C2Classifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _classifier

    logger.info("Loading corpus from %s...", DATA_DIR)
    docs = load_corpus(DATA_DIR)
    logger.info("Corpus: %d docs loaded.", len(docs))

    # Build similarity index
    embedder = get_embedder(EMBEDDING_METHOD)
    _index = CorpusIndex(embedder)
    _index.build(docs)

    # Load or train C2 classifier
    _classifier = C2Classifier()
    _classifier.load_or_train(docs, MODEL_CACHE)

    logger.info("Service ready. Embedding: %s | Corpus: %d sentences",
                EMBEDDING_METHOD, _index.size)
    logger.info("=" * 60)
    logger.info("  API docs  →  http://localhost:8000/docs")
    logger.info("  Web UI    →  http://localhost:8501")
    logger.info("=" * 60)
    yield

    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ToS Semantic Similarity & Classification API",
    description="Semantic search and unfair clause detection for Terms of Service.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_index() -> CorpusIndex:
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not ready.")
    return _index


def _get_classifier() -> C2Classifier:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not ready.")
    return _classifier


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    idx = _get_index()
    return HealthResponse(
        status="ok",
        embedding_method=EMBEDDING_METHOD,
        corpus_size=idx.size,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    idx = _get_index()
    results = idx.search(req.query, req.top_k)
    return SearchResponse(results=results, embedding_method=EMBEDDING_METHOD)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    clf = _get_classifier()
    result = clf.predict(req.sentence)
    return ClassifyResponse(**result)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    idx = _get_index()
    clf = _get_classifier()

    classification_dict = clf.predict(req.sentence)
    classification = ClassifyResponse(**classification_dict)
    similar = idx.search(req.sentence, req.top_k)

    return AnalyzeResponse(classification=classification, similar=similar)
