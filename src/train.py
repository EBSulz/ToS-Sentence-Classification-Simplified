import numpy as np
from sklearn.model_selection import KFold

from .data_loader import CATEGORIES
from .preprocessing import clean
from .features import make_vectorizer
from .models import make_model
from .evaluate import doc_metrics

N_SPLITS = 5


def _preprocess_all(docs: list[dict]) -> list[list[str]]:
    return [[clean(s) for s in d['sentences']] for d in docs]


def _cv_splits(n_docs: int) -> list[tuple]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    return list(kf.split(range(n_docs)))


def _compute_spw(y: np.ndarray) -> float:
    """XGBoost scale_pos_weight = negatives / positives."""
    pos = int((y == 1).sum())
    neg = int((y != 1).sum())
    return neg / pos if pos > 0 else 9.0


def _fit_model(model_name: str, X_train, y_train: np.ndarray):
    """Fit and return a model. Called once per CV fold."""
    model = make_model(model_name, scale_pos_weight=_compute_spw(y_train))
    y_fit = (y_train == 1).astype(int) if model_name == 'XGBoost' else y_train
    model.fit(X_train, y_fit)
    return model


def _decode(model_name: str, model, X_test) -> np.ndarray:
    """Predict and normalise to -1/1."""
    pred = model.predict(X_test)
    if model_name == 'XGBoost':
        pred = np.where(pred == 1, 1, -1)
    return pred.astype(int)


# ── Standard classifiers (C1 / C2 style) ─────────────────────────────────────

def run_c1(docs: list[dict], model_name: str) -> list[tuple[float, float, float]]:
    """
    C1: single classifier on general unfairness labels.
    5-fold document-level CV; model fitted once per fold.
    """
    cleaned = _preprocess_all(docs)
    splits = _cv_splits(len(docs))
    results: list[tuple[float, float, float]] = []

    for train_idx, test_idx in splits:
        train_sents = [s for i in train_idx for s in cleaned[i]]
        y_train = np.array([l for i in train_idx for l in docs[i]['labels']])

        vec = make_vectorizer()
        X_train = vec.fit_transform(train_sents)
        model = _fit_model(model_name, X_train, y_train)

        for i in test_idx:
            y_test = np.array(docs[i]['labels'])
            if (y_test == 1).sum() == 0:
                continue
            X_test = vec.transform(cleaned[i])
            results.append(doc_metrics(y_test, _decode(model_name, model, X_test)))

    return results


def run_c2(docs: list[dict], model_name: str) -> list[tuple[float, float, float]]:
    """
    C2: one classifier per category; unfair if ANY category fires (OR rule).
    Shared vectorizer per fold; evaluated against general labels.
    """
    cleaned = _preprocess_all(docs)
    splits = _cv_splits(len(docs))
    results: list[tuple[float, float, float]] = []

    for train_idx, test_idx in splits:
        train_sents = [s for i in train_idx for s in cleaned[i]]

        vec = make_vectorizer()
        X_train = vec.fit_transform(train_sents)

        cat_models: dict = {}
        for cat in CATEGORIES:
            y_cat = np.array([l for i in train_idx for l in docs[i]['cat_labels'][cat]])
            if (y_cat == 1).sum() == 0:
                continue
            cat_models[cat] = _fit_model(model_name, X_train, y_cat)

        for i in test_idx:
            y_test = np.array(docs[i]['labels'])
            if (y_test == 1).sum() == 0:
                continue
            X_test = vec.transform(cleaned[i])
            combined = np.full(len(cleaned[i]), -1, dtype=int)
            for cat, model in cat_models.items():
                combined = np.where(_decode(model_name, model, X_test) == 1, 1, combined)
            results.append(doc_metrics(y_test, combined))

    return results
