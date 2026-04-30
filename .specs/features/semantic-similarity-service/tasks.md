# Semantic Similarity Service Tasks

**Design**: `.specs/features/semantic-similarity-service/design.md`
**Status**: In Progress

---

## Execution Plan

```
Phase 1 — Foundation (parallel):
  T1 [P] ──────────────────────────────────────────────────────┐
  T2 [P] ──────────────────────────────────────────────────────┤
  T3 [P] ──────────────────────────────────────────────────────┤
  T9 [P] ──────────────────────────────────────────────────────┘
                    ↓ all complete
Phase 2 — Index + Experiment (parallel):
  T4 [P] (depends T1, T2) ─────────────────────────────────────┐
  T8 [P] (depends T2) ─────────────────────────────────────────┘
                    ↓ all complete
Phase 3 — API (sequential):
  T5 (depends T1, T3, T4)
                    ↓
Phase 4 — UI + Docker (parallel):
  T6 [P] (depends T5) ─────────────────────────────────────────┐
  T7 [P] (depends T5) ─────────────────────────────────────────┘
```

---

## Task Breakdown

### T1: Create service/schemas.py + service/__init__.py [P]

**What**: Pydantic v2 schemas for all API request/response types + empty package init
**Where**: `service/__init__.py`, `service/schemas.py`
**Depends on**: None
**Reuses**: Nothing
**Requirement**: SIM-01, CLS-01, ANA-01, ANA-02

**Done when**:
- [ ] `service/__init__.py` exists (empty)
- [ ] `SearchRequest`, `SearchResult`, `SearchResponse` defined
- [ ] `ClassifyRequest`, `ClassifyResponse` defined
- [ ] `AnalyzeRequest`, `AnalyzeResponse` defined
- [ ] `HealthResponse` defined
- [ ] `python -c "from service.schemas import AnalyzeResponse"` exits 0

---

### T2: Create service/embeddings.py [P]

**What**: `BaseEmbedder` ABC + `TFIDFEmbedder` + `SBERTEmbedder` + `get_embedder()` factory
**Where**: `service/embeddings.py`
**Depends on**: None (reuses `src.preprocessing`, `src.features` which already exist)
**Reuses**: `clean()` from `src/preprocessing.py`, `make_vectorizer()` from `src/features.py`
**Requirement**: SIM-01, EXP-01

**Done when**:
- [ ] `BaseEmbedder` ABC with `fit(sentences)`, `encode(sentences) -> np.ndarray`, `name: str`
- [ ] `TFIDFEmbedder`: calls `clean()` on each sentence, fits `make_vectorizer()`, returns L2-normalized dense array
- [ ] `SBERTEmbedder`: wraps `SentenceTransformer('all-MiniLM-L6-v2')`, returns L2-normalized array
- [ ] `get_embedder('tfidf')` and `get_embedder('sbert')` return correct instances
- [ ] `python -c "from service.embeddings import get_embedder; e=get_embedder('tfidf')"` exits 0

---

### T3: Create service/classifier.py [P]

**What**: `C2Classifier` — trains 8 LinearSVC on full corpus, persists with joblib, predicts per sentence
**Where**: `service/classifier.py`
**Depends on**: None (reuses existing `src/` modules)
**Reuses**: `load_corpus()`, `clean()`, `make_vectorizer()`, `make_model('SVM')`, `CATEGORIES`
**Requirement**: CLS-01, CLS-02

**Done when**:
- [ ] `C2Classifier` class with `train(docs)`, `save(cache_dir)`, `load(cache_dir)`, `load_or_train(docs, cache_dir)`, `predict(sentence) -> ClassifyResponse`
- [ ] `train()` skips categories with 0 positives (no crash)
- [ ] `predict()` returns correct `ClassifyResponse` structure
- [ ] `python -c "from service.classifier import C2Classifier"` exits 0

---

### T4: Create service/corpus_index.py [P]

**What**: `CorpusIndex` — flat corpus list + embedding matrix + cosine top-k search
**Where**: `service/corpus_index.py`
**Depends on**: T1 (schemas), T2 (embeddings)
**Reuses**: `SearchResult` from `service/schemas.py`, `BaseEmbedder` from `service/embeddings.py`
**Requirement**: SIM-01, SIM-02, SIM-03

**Done when**:
- [ ] `CorpusIndex(embedder)` class with `build(docs)` and `search(query, top_k) -> list[SearchResult]`
- [ ] `build()` stores flat `(sentence, doc, is_unfair)` tuples and embedding matrix
- [ ] `search()` preprocesses query (via embedder), computes cosine similarity, returns sorted top-k
- [ ] `search()` clamps top_k to corpus size
- [ ] `python -c "from service.corpus_index import CorpusIndex"` exits 0

---

### T5: Create service/api.py

**What**: FastAPI app with `/health`, `/search`, `/classify`, `/analyze` endpoints and lifespan startup
**Where**: `service/api.py`
**Depends on**: T1, T3, T4
**Reuses**: All `service/` modules; `load_corpus()` from `src/data_loader.py`
**Requirement**: SIM-01, CLS-01, ANA-01, ANA-02, DOC-02

**Done when**:
- [ ] `GET /health` returns `HealthResponse`
- [ ] `POST /search` calls `corpus_index.search()` and returns `SearchResponse`
- [ ] `POST /classify` calls `classifier.predict()` and returns `ClassifyResponse`
- [ ] `POST /analyze` calls both and returns `AnalyzeResponse`
- [ ] Lifespan: loads corpus → builds index → calls `load_or_train()` → logs ready
- [ ] `DATA_DIR`, `MODEL_CACHE`, `EMBEDDING_METHOD` env vars with defaults
- [ ] `python -c "from service.api import app"` exits 0

---

### T6: Create app.py (Streamlit UI) [P]

**What**: Streamlit frontend — text area, analyze button, classification badge, similar sentences table
**Where**: `app.py`
**Depends on**: T5 (API contract defined)
**Reuses**: API schemas (for expected JSON shapes)
**Requirement**: UI-01, UI-02

**Done when**:
- [ ] Sidebar: API_URL input (default `http://localhost:8000`), top_k slider
- [ ] Main: `st.text_area` + "Analyze" button
- [ ] On submit: calls `POST /analyze`, displays `is_unfair` badge and categories
- [ ] Similar sentences shown in `st.dataframe`
- [ ] On `httpx.RequestError`: `st.error("Cannot reach API at {url}")`
- [ ] `API_URL` env var sets default URL (for Docker use)

---

### T7: Create Dockerfile + docker-compose.yaml [P]

**What**: Single Dockerfile (API entry) + docker-compose.yaml with api + app services
**Where**: `Dockerfile`, `docker-compose.yaml`
**Depends on**: T5, T6
**Reuses**: `requirements.txt` (updated in T9)
**Requirement**: DOC-01, DOC-02

**Done when**:
- [ ] `Dockerfile` uses `python:3.11-slim`, installs requirements, downloads NLTK stopwords at build, sets `WORKDIR /app`
- [ ] API service in compose: port `8000:8000`, mounts `./ToS:/app/ToS` and `./data:/app/data`
- [ ] App service in compose: port `8501:8501`, cmd `streamlit run app.py --server.address 0.0.0.0`, env `API_URL=http://api:8000`, depends_on `api`
- [ ] `docker build -t tos-service .` succeeds (syntax valid)

---

### T8: Create scripts/embedding_experiment.py [P]

**What**: Compare TF-IDF vs SBERT retrieval quality; print P@5, P@10, MAP comparison table
**Where**: `scripts/__init__.py`, `scripts/embedding_experiment.py`
**Depends on**: T2, T4
**Reuses**: `service/embeddings.py`, `service/corpus_index.py`, `src/data_loader.py`
**Requirement**: EXP-01

**Done when**:
- [ ] 80/20 document split (40 train docs, 10 test docs, random seed 42)
- [ ] For each method: build index on train, query each test sentence that is unfair
- [ ] Computes P@5, P@10, MAP for each method
- [ ] Prints ASCII comparison table
- [ ] `python scripts/embedding_experiment.py` runs without error (may take minutes for SBERT)

---

### T9: Update requirements.txt [P]

**What**: Add all new runtime dependencies
**Where**: `requirements.txt`
**Depends on**: None
**Requirement**: DOC-01

**Done when**:
- [ ] `fastapi>=0.110.0` added
- [ ] `uvicorn[standard]>=0.27.0` added
- [ ] `sentence-transformers>=2.6.0` added
- [ ] `httpx>=0.27.0` added
- [ ] `streamlit>=1.32.0` added
- [ ] `joblib>=1.3.0` added
- [ ] `pandas>=2.0.0` added
- [ ] `pydantic>=2.0.0` added

---

## Parallel Execution Map

```
Phase 1 (all parallel — no dependencies):
  T1 [P]  T2 [P]  T3 [P]  T9 [P]

Phase 2 (parallel after Phase 1):
  T4 [P] (needs T1+T2)
  T8 [P] (needs T2+T4 — actually T4 needs T2, so T8 waits for T4)

Actually T8 depends on T4, so:
Phase 2a:  T4 [P] (needs T1, T2)
Phase 2b:  T8 [P] (needs T4)

Phase 3 (sequential):
  T5 (needs T1, T3, T4)

Phase 4 (parallel after T5):
  T6 [P]  T7 [P]
```

---

## Granularity Check

| Task | Scope | Status |
|---|---|---|
| T1: schemas + init | 2 files, pure data models | ✅ Granular |
| T2: embeddings.py | 1 file, 2 classes + factory | ✅ Granular |
| T3: classifier.py | 1 file, 1 class | ✅ Granular |
| T4: corpus_index.py | 1 file, 1 class | ✅ Granular |
| T5: api.py | 1 file, 4 endpoints | ✅ Granular |
| T6: app.py | 1 file, UI only | ✅ Granular |
| T7: Dockerfile + compose | 2 config files, same concern | ✅ Granular |
| T8: experiment script | 1 file | ✅ Granular |
| T9: requirements.txt | 1 file, add deps | ✅ Granular |

## Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram Shows | Status |
|---|---|---|---|
| T1 | None | Phase 1 parallel | ✅ |
| T2 | None | Phase 1 parallel | ✅ |
| T3 | None | Phase 1 parallel | ✅ |
| T9 | None | Phase 1 parallel | ✅ |
| T4 | T1, T2 | Phase 2 after Phase 1 | ✅ |
| T8 | T2, T4 | Phase 2b after T4 | ✅ |
| T5 | T1, T3, T4 | Phase 3 sequential | ✅ |
| T6 | T5 | Phase 4 parallel | ✅ |
| T7 | T5, T6 | Phase 4 parallel | ✅ |
