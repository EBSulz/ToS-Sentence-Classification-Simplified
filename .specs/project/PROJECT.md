# ToS Sentence Classification

## Vision
Detect potentially unfair clauses in Terms of Service documents using ML.

## Goals
- Reproduce C1 (single general-label classifier) and C2 (8 category classifiers OR-combined) from CLAUDETTE paper
- Test SVM, Logistic Regression, XGBoost on BoW (TF-IDF unigrams+bigrams)
- Report macro-averaged P, R, F1 via LOO evaluation (50 folds)

## Stack
- Python 3.10+
- scikit-learn (SVM, LR, TF-IDF)
- xgboost
- nltk (stopwords)

## Data
- 50 ToS documents, ~9-12k sentences
- 8 unfairness categories: a, ch, cr, j, law, ltd, ter, use
- Labels: 1=unfair, -1=fair
- Preprocessing already done (tokenized, lowercased, Penn Treebank format)
