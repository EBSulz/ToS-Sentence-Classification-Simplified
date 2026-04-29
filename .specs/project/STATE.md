# State

## Decisions
- LOO (leave-one-document-out): skip test docs with zero positive labels (undefined metrics)
- XGBoost: labels converted 0/1 internally; scale_pos_weight computed per fold
- SVM/LR: class_weight='balanced', labels -1/1
- TF-IDF: ngram_range=(1,2), sublinear_tf=True, min_df=2, max_features=50000
- C2 OR logic: predict unfair if ANY of 8 category classifiers fires

## Preferences
- Lightweight models preferred for session updates
