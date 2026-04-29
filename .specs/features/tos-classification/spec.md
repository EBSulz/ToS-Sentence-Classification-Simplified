# Feature Spec: ToS Sentence Classification (BoW Baseline)

## Requirements

### R1 - Data Loading
- Load all 50 ToS sentence files (line-aligned with label files)
- Load general labels from Labels/
- Load 8 category labels from Labels_A/ through Labels_USE/
- Files already tokenized/lowercased (PTB format); handle gracefully if category file missing

### R2 - Preprocessing
- Remove PTB bracket tokens (-lrb-, -rrb-, etc.)
- Remove NLTK English stopwords
- Remove punctuation-only tokens
- Output: cleaned sentence string for TF-IDF

### R3 - Feature Extraction
- TF-IDF with unigrams+bigrams (ngram_range=(1,2))
- sublinear_tf=True, min_df=2
- Vectorizer fit on training split only (per LOO fold) — no leakage

### R4 - C1: Single Classifier
- Binary classification: 1=unfair, -1=fair (general label)
- LOO evaluation: 50 folds, skip folds with no positive test examples
- Report macro-averaged P, R, F1

### R5 - C2: Combined Classifiers
- 8 binary classifiers, one per category (same features)
- Sentence predicted unfair = OR of all 8 predictions
- Evaluate against general labels, same LOO + metrics

### R6 - Models
- SVM (LinearSVC, class_weight='balanced')
- Logistic Regression (class_weight='balanced')
- XGBoost (scale_pos_weight auto-computed per fold)

### R7 - Output
- Print macro-averaged P / R / F1 per model per experiment (C1, C2)

## Done When
- main.py runs end-to-end, prints results table for C1 and C2 × 3 models
