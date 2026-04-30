# Semantic Similarity Service Design

**Spec**: `.specs/features/semantic-similarity-service/spec.md`
**Status**: Approved

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  docker-compose                                          │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │   app (Streamlit)│─HTTP───▶│   api (FastAPI)      │  │
│  │   :8501          │         │   :8000              │  │
│  └──────────────────┘         │                      │  │
│                               │  ┌────────────────┐  │  │
│                               │  │ CorpusIndex    │  │  │
│                               │  │ (embeddings +  │  │  │
│                               │  │  cosine search)│  │  │
│                               │  └────────────────┘  │  │
│                               │  ┌────────────────┐  │  │
│                               │  │ C2Classifier   │  │  │
│                               │  │ (8 SVM models) │  │  │
│                               │  └────────────────┘  │  │
│                               └──────────┬───────────┘  │
│                                          │              │
│                               ┌──────────▼───────────┐  │
│                               │   ToS/ data volume   │  │
│                               │   data/ model cache  │  │
│                               └──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Request flow for `/analyze`:
```
User (Streamlit) → POST /analyze → CorpusIndex.search() + C2Classifier.predict()
                                 → SearchResult[] + ClassificationResult
                 ← Response JSON
```

---

## Code Reuse Analysis

### Existing Components to Leverage

| Component | Location | How to Use |
|---|---|---|
| `load_corpus()` | `src/data_loader.py` | Load all 50 docs for indexing + classifier training |
| `clean()` | `src/preprocessing.py` | Preprocess queries + corpus sentences for TF-IDF |
| `make_vectorizer()` | `src/features.py` | TF-IDF vectorizer for TFIDFEmbedder + C2Classifier |
| `LinearSVC` + `make_model()` | `src/models.py` | C2 SVM models (use `make_model('SVM')` per category) |
| `CATEGORIES` | `src/data_loader.py` | List of 8 category keys |

### Integration Points

| System | Integration Method |
|---|---|
| Existing `src/` modules | Import directly in `service/` modules |
| ToS data | Mount as volume; path configurable via `DATA_DIR` env var |
| Model cache | `data/models/` dir; created if absent; joblib pickle |

---

## Components

### `service/schemas.py`

- **Purpose**: Pydantic v2 request/response models for all API endpoints
- **Location**: `service/schemas.py`
- **Interfaces**:
  - `SearchRequest(query: str, top_k: int = 10)`
  - `SearchResult(sentence: str, doc: str, score: float, is_unfair: bool)`
  - `SearchResponse(results: list[SearchResult], embedding_method: str)`
  - `ClassifyRequest(sentence: str)`
  - `ClassifyResponse(is_unfair: bool, categories: list[str], details: dict[str, bool])`
  - `AnalyzeRequest(sentence: str, top_k: int = 10)`
  - `AnalyzeResponse(classification: ClassifyResponse, similar: list[SearchResult])`
  - `HealthResponse(status: str, embedding_method: str, corpus_size: int)`
- **Dependencies**: `pydantic`
- **Reuses**: Nothing — pure data models

---

### `service/embeddings.py`

- **Purpose**: Two interchangeable sentence encoders used by CorpusIndex
- **Location**: `service/embeddings.py`
- **Interfaces**:
  - `BaseEmbedder` (ABC): `fit(sentences)`, `encode(sentences) -> np.ndarray`, `name: str`
  - `TFIDFEmbedder(BaseEmbedder)`: wraps `make_vectorizer()` + `clean()`; returns dense L2-normalized vectors
  - `SBERTEmbedder(BaseEmbedder)`: wraps `sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')`; returns L2-normalized vectors
  - `get_embedder(method: str) -> BaseEmbedder`: factory, `method ∈ {"tfidf", "sbert"}`
- **Dependencies**: `src.preprocessing`, `src.features`, `sentence_transformers`, `numpy`
- **Reuses**: `clean()`, `make_vectorizer()`

---

### `service/corpus_index.py`

- **Purpose**: Loads corpus, builds embedding matrix, answers top-k similarity queries
- **Location**: `service/corpus_index.py`
- **Interfaces**:
  - `CorpusIndex(embedder: BaseEmbedder)`
  - `.build(docs: list[dict]) -> None` — fits embedder on all sentences, builds matrix
  - `.search(query: str, top_k: int) -> list[SearchResult]` — cosine similarity, sorted descending
- **Dependencies**: `service/embeddings.py`, `service/schemas.py`, `numpy`, `scipy`
- **Reuses**: `load_corpus()` (called externally; index receives docs)
- **Implementation note**: Store flat list of `(sentence, doc_name, is_unfair)` tuples + embedding matrix as `np.ndarray`. Cosine search = normalize then dot product.

---

### `service/classifier.py`

- **Purpose**: Trains/loads C2 SVM (8 category classifiers, OR-combined), predicts on a single sentence
- **Location**: `service/classifier.py`
- **Interfaces**:
  - `C2Classifier()`
  - `.train(docs: list[dict]) -> None` — trains 8 LinearSVC on full corpus
  - `.save(cache_dir: Path) -> None` — joblib dump vectorizer + models dict
  - `.load(cache_dir: Path) -> None` — joblib load; sets `_ready = True`
  - `.load_or_train(docs, cache_dir) -> None` — convenience: load if cache exists, else train + save
  - `.predict(sentence: str) -> ClassifyResponse` — preprocess → TF-IDF → 8 predict → OR → response
- **Dependencies**: `src.data_loader`, `src.preprocessing`, `src.features`, `src.models`, `joblib`
- **Reuses**: `clean()`, `make_vectorizer()`, `make_model('SVM')`, `CATEGORIES`
- **Cache files**: `data/models/c2_vectorizer.joblib`, `data/models/c2_models.joblib`

---

### `service/api.py`

- **Purpose**: FastAPI application wiring all components; loads everything on startup
- **Location**: `service/api.py`
- **Interfaces**:
  - `GET /health` → `HealthResponse`
  - `POST /search` → `SearchResponse`
  - `POST /classify` → `ClassifyResponse`
  - `POST /analyze` → `AnalyzeResponse`
- **Dependencies**: `fastapi`, `uvicorn`, `service/corpus_index`, `service/classifier`, `service/schemas`
- **Startup**: `lifespan` handler loads corpus → builds index → loads/trains classifier
- **Config**: `DATA_DIR` (default `ToS`), `MODEL_CACHE` (default `data/models`), `EMBEDDING_METHOD` (default `tfidf`), `TOP_K_MAX` (default `50`)
- **Reuses**: `load_corpus()`

---

### `app.py`

- **Purpose**: Streamlit frontend; calls FastAPI via `httpx`
- **Location**: `app.py` (project root)
- **Interfaces**: Web UI only — no Python API
- **Layout**:
  - Sidebar: API URL (default `http://localhost:8000`), top-k slider (1–30, default 10)
  - Main: large text area + "Analyze" button
  - Results: classification badge (🔴 Unfair / 🟢 Fair) + categories chips + similar sentences table
- **Dependencies**: `streamlit`, `httpx`, `pandas`
- **Reuses**: None — standalone UI

---

### `scripts/embedding_experiment.py`

- **Purpose**: Compare TF-IDF cosine vs SBERT on retrieval quality; print results table
- **Location**: `scripts/embedding_experiment.py`
- **Evaluation**:
  - 80/20 document-level split (40 train, 10 test)
  - For each method: build index on train sentences, query with all test sentences
  - Metrics (computed over unfair query sentences only): P@5, P@10, MAP
- **Output**: ASCII table comparing both methods
- **Dependencies**: `service/embeddings`, `service/corpus_index`, `src/data_loader`, `numpy`

---

### `Dockerfile`

- Base: `python:3.11-slim`
- Install: `requirements.txt`
- NLTK data download at build time
- Entrypoint: `uvicorn service.api:app --host 0.0.0.0 --port 8000`
- Separate Streamlit target via docker-compose command override

### `docker-compose.yaml`

- `api`: builds image, port `8000:8000`, mounts `./ToS:/app/ToS` and `./data:/app/data`, env `EMBEDDING_METHOD`
- `app`: same image, port `8501:8501`, command `streamlit run app.py`, depends on `api`, env `API_URL=http://api:8000`

---

## Data Models

```python
# In-memory corpus entry (internal)
CorpusEntry = namedtuple('CorpusEntry', ['sentence', 'doc', 'is_unfair'])

# Embedding matrix shape: (N_sentences, embedding_dim)
# TF-IDF: (9400, ≤50000) — stored dense after normalization
# SBERT: (9400, 384) — all-MiniLM-L6-v2 output dim
```

---

## Error Handling Strategy

| Error Scenario | Handling | User Impact |
|---|---|---|
| Empty query | Pydantic `min_length=1` validator | 422 Unprocessable Entity |
| top_k > corpus size | `min(top_k, len(corpus))` | Silent clamp, still works |
| API unreachable (Streamlit) | `httpx.RequestError` catch → `st.error()` | Red error banner |
| SBERT model not cached | Auto-download on first start | ~15s delay, logged |
| Category with 0 positives | Skipped in training, always predicts -1 | Never fires for that category |

---

## Tech Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Similarity search backend | numpy cosine (brute force) | 9,400 sentences × 384 dims fits comfortably in RAM; no FAISS overhead |
| Default embedding | TF-IDF | Fast startup, no model download, good baseline for legal text |
| SVM model for inference | C2 SVM (8 LinearSVC, OR-combined) | Best single-method F1 in paper experiments |
| Model persistence | joblib | Already a sklearn dependency; simple, fast |
| API framework | FastAPI | Pydantic validation, async lifespan, auto OpenAPI docs |
| Frontend | Streamlit | Minimal code, data-science friendly, no JS required |
