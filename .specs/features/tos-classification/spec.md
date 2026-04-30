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

### R4 - Evaluation: 5-fold document-level CV
- Split 50 documents into 5 folds (10 docs each); train on 40, test on 10
- Model + vectorizer fitted ONCE per fold (no per-document refit)
- Skip test docs with zero positive examples
- Macro-average P, R, F1 across all test documents

### R5 - C1: Single Classifier
- Binary classification: 1=unfair, -1=fair (general label)
- Models: SVM, LR, RandomForest, XGBoost

### R6 - C2: Combined Classifiers
- 8 binary classifiers, one per category (shared vectorizer)
- Sentence predicted unfair = OR of all 8 predictions
- Evaluate against general labels

### R7 - C6: SVM-HMM Single Model
- CalibratedClassifierCV(LinearSVC) for emission probabilities
- HMM transition probs estimated from training label sequences
- Viterbi decoding per test document (exploits sentence order)

### R8 - C7: SVM-HMM 8 Category Classifiers
- Same as C6 but one SVMHMM per category, OR-combined

### R9 - Models
- SVM, LogisticRegression, RandomForest, XGBoost (for C1/C2)
- SVMHMM (for C6/C7)

### R10 - Output
- Print macro-averaged P / R / F1 per model for C1, C2, C6, C7

## Done When
- main.py runs end-to-end, prints results for C1/C2 (×4 models) + C6 + C7
