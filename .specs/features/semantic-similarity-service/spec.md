# Semantic Similarity Service Specification

**Status**: Approved

## Problem Statement

The ToS classification pipeline can detect unfair clauses but is not consumable as a service. Users need a way to submit arbitrary ToS text and (1) find the most semantically similar sentences from the labeled corpus, and (2) immediately see whether that text is flagged as a potentially unfair clause and which category. This requires a REST API, a trained inference model, and a web UI — all containerized.

## Goals

- [ ] Index all ~9,400 corpus sentences and support semantic similarity search via REST API
- [ ] Expose the best C2 SVM classifier (8 category, OR-combined) for real-time inference
- [ ] Compare TF-IDF cosine vs sentence-transformers as embedding methods (experiment)
- [ ] Ship a Streamlit UI and Docker setup to run the whole stack

## Out of Scope

| Feature | Reason |
|---|---|
| Generative LLMs / RAG | Explicitly prohibited by task rules |
| Re-running full CV experiments via API | Classification comparison already done in main.py |
| User auth / rate limiting | Out of scope for MVP |
| Persistent vector DB (FAISS, Pinecone) | Corpus is small (~9,400); numpy cosine is fast enough |

---

## User Stories

### P1: Semantic similarity search ⭐ MVP

**User Story**: As a researcher, I want to submit a sentence and receive the most similar sentences from the ToS corpus so that I can find precedents for clauses.

**Acceptance Criteria**:

1. WHEN `POST /search` receives `{query, top_k}` THEN system SHALL return top_k results sorted by similarity score descending
2. WHEN a result is returned THEN each result SHALL include: sentence text, document name, similarity score, and whether the sentence is labeled unfair in the corpus
3. WHEN query is empty or whitespace THEN system SHALL return 422 validation error

**Independent Test**: `curl -X POST /search -d '{"query":"we may terminate your account at any time","top_k":5}'` returns 5 sentences with scores.

---

### P1: Unfair clause classification ⭐ MVP

**User Story**: As a researcher, I want to classify a sentence and see which unfairness categories it triggers so that I can understand what makes a clause problematic.

**Acceptance Criteria**:

1. WHEN `POST /classify` receives `{sentence}` THEN system SHALL return `is_unfair: bool`, `categories: list[str]`, and `details: dict[str, bool]` for all 8 categories
2. WHEN `is_unfair` is true THEN `categories` SHALL be non-empty and list only the categories that fired
3. WHEN sentence is shorter than 5 words THEN system SHALL still return a result (classifier handles it)

**Independent Test**: Submitting a known unfair sentence from the corpus returns `is_unfair: true` with at least one category.

---

### P1: Combined analyze endpoint ⭐ MVP

**User Story**: As a researcher, I want a single endpoint that both classifies and finds similar sentences so that I get full context in one call.

**Acceptance Criteria**:

1. WHEN `POST /analyze` receives `{sentence, top_k}` THEN system SHALL return both `classification` and `similar` fields
2. WHEN the service is healthy THEN `GET /health` SHALL return `{status, embedding_method, corpus_size}`

**Independent Test**: `/analyze` response has both `classification` and `similar` keys.

---

### P1: Streamlit web UI ⭐ MVP

**User Story**: As a user without API knowledge, I want a simple web form where I paste a ToS sentence and immediately see similarity results and classification so that I don't need to use curl.

**Acceptance Criteria**:

1. WHEN user enters text and clicks analyze THEN UI SHALL display: is_unfair badge, flagged categories, and a table of similar sentences
2. WHEN API is unreachable THEN UI SHALL display a clear error message
3. WHEN top_k slider is adjusted THEN search results SHALL update accordingly

**Independent Test**: Open `http://localhost:8501`, paste a sentence, click analyze, see results.

---

### P2: Embedding experiment script

**User Story**: As a researcher, I want a script that compares TF-IDF cosine vs sentence-transformers on retrieval quality so that I can justify the embedding choice in my report.

**Acceptance Criteria**:

1. WHEN `python scripts/embedding_experiment.py` is run THEN it SHALL print P@5, P@10, and MAP for both embedding methods
2. WHEN experiment completes THEN results SHALL be computed using document-level held-out evaluation (80/20 split by document)

**Independent Test**: Script runs to completion and prints a comparison table.

---

### P1: Docker containerization ⭐ MVP

**User Story**: As a reviewer, I want to run the full stack with a single `docker-compose up` so that I don't need to manually install dependencies.

**Acceptance Criteria**:

1. WHEN `docker-compose up` is run THEN API SHALL be reachable at `localhost:8000` and Streamlit at `localhost:8501`
2. WHEN the API container starts THEN it SHALL train/load models and index the corpus before serving requests
3. WHEN models are already cached THEN startup SHALL skip retraining

**Independent Test**: `docker-compose up` + `curl localhost:8000/health` returns 200.

---

## Edge Cases

- WHEN corpus sentence has fewer than 5 words THEN it is still indexed (already in corpus)
- WHEN a category has no positive labels in training data THEN that category's classifier is skipped and never fires
- WHEN `EMBEDDING_METHOD=sbert` and model not downloaded THEN model is downloaded on first startup
- WHEN `top_k` exceeds corpus size THEN return all sentences

---

## Requirement Traceability

| Requirement ID | Story | Status |
|---|---|---|
| SIM-01 | P1: Similarity search | Pending |
| SIM-02 | P1: Similarity search | Pending |
| SIM-03 | P1: Similarity search | Pending |
| CLS-01 | P1: Classification | Pending |
| CLS-02 | P1: Classification | Pending |
| ANA-01 | P1: Analyze endpoint | Pending |
| ANA-02 | P1: Health endpoint | Pending |
| UI-01 | P1: Streamlit UI | Pending |
| UI-02 | P1: Streamlit UI | Pending |
| EXP-01 | P2: Embedding experiment | Pending |
| DOC-01 | P1: Docker | Pending |
| DOC-02 | P1: Docker | Pending |

---

## Success Criteria

- [ ] `docker-compose up` → both services healthy in < 2 min
- [ ] `/analyze` correctly flags known unfair sentences from corpus (spot-check)
- [ ] Embedding experiment script prints results for both methods
- [ ] Streamlit UI loads and returns results without errors
