# State

## Decisions
- Evaluation: 5-fold document-level CV (KFold shuffle=True seed=42); replaces LOO
  - Skip test docs with zero positive labels (F1 undefined)
  - Model fitted ONCE per fold (not per test doc); vectorizer also once per fold
- XGBoost: labels converted 0/1 internally; scale_pos_weight = neg/pos computed per fold
- SVM/LR/RF: class_weight='balanced', labels -1/1
- TF-IDF: ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=50000
- C2 OR logic: predict unfair if ANY of 8 category classifiers fires
## Models
- C1/C2: SVM, LogisticRegression, RandomForest, XGBoost

## Service Layer (semantic-similarity-service feature)
- FastAPI service: `service/api.py` — uvicorn, lifespan startup loads corpus+index+classifier
- Embedding methods: `tfidf` (default, fast) and `sbert` (all-MiniLM-L6-v2); set via EMBEDDING_METHOD env var
- Similarity search: brute-force cosine (numpy dot on L2-normalised vecs); no FAISS needed at ~9,400 sentences
- Inference classifier: C2 SVM (8 LinearSVC, OR-combined) trained on FULL corpus at startup; cached with joblib in data/models/
- Streamlit UI: `app.py` — calls /analyze; API_URL env var for Docker
- Docker: single Dockerfile (python:3.11-slim); docker-compose with `api:8000` + `app:8501`; app waits for api healthcheck
- Experiment: `scripts/embedding_experiment.py` — 80/20 doc split, P@5/P@10/MAP for tfidf vs sbert

## Preferences
- Lightweight models preferred for session updates
