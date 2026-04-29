from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF vectorizer with unigrams + bigrams, matching paper (C1/C2).
    Always fit on training data only — never on the test fold.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,   # log(1 + tf) dampens high-frequency terms
        min_df=2,            # ignore terms that appear in fewer than 2 docs
        max_features=50_000,
    )
