import numpy as np
from pathlib import Path

from .data_loader import CATEGORIES
from .preprocessing import clean
from .features import make_vectorizer
from .models import make_model
from .evaluate import doc_metrics, macro_average


def _preprocess_all(docs: list[dict]) -> list[list[str]]:
    return [[clean(s) for s in d['sentences']] for d in docs]


def _compute_spw(y_train: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight = negatives / positives."""
    pos = int((y_train == 1).sum())
    neg = int((y_train != 1).sum())
    return neg / pos if pos > 0 else 9.0


def _fit_predict(
    model_name: str,
    X_train,
    y_train: np.ndarray,
    X_test,
) -> np.ndarray:
    """
    Fit model on training data and return predictions on test data.
    Labels expected as -1/1.  XGBoost internally uses 0/1.
    Returned predictions are always -1/1.
    """
    spw = _compute_spw(y_train)
    model = make_model(model_name, scale_pos_weight=spw)

    if model_name == 'XGBoost':
        y_fit = (y_train == 1).astype(int)
    else:
        y_fit = y_train

    model.fit(X_train, y_fit)
    pred = model.predict(X_test)

    if model_name == 'XGBoost':
        pred = np.where(pred == 1, 1, -1)

    return pred.astype(int)


def run_c1(docs: list[dict], model_name: str) -> list[tuple[float, float, float]]:
    """
    C1: single binary classifier on general unfairness labels.
    LOO evaluation — returns per-fold (P, R, F1) tuples.
    """
    cleaned = _preprocess_all(docs)
    fold_results: list[tuple[float, float, float]] = []

    for i, doc in enumerate(docs):
        y_test = np.array(doc['labels'])
        if (y_test == 1).sum() == 0:
            continue  # skip: F1 undefined when no positives in test doc

        train_sents: list[str] = [
            s for j, sents in enumerate(cleaned) if j != i for s in sents
        ]
        train_labels: list[int] = [
            lbl for j, d in enumerate(docs) if j != i for lbl in d['labels']
        ]
        y_train = np.array(train_labels)

        vec = make_vectorizer()
        X_train = vec.fit_transform(train_sents)
        X_test = vec.transform(cleaned[i])

        pred = _fit_predict(model_name, X_train, y_train, X_test)
        fold_results.append(doc_metrics(y_test, pred))

    return fold_results


def run_c2(docs: list[dict], model_name: str) -> list[tuple[float, float, float]]:
    """
    C2: one classifier per category; sentence is unfair if ANY category fires.
    Same LOO evaluation as C1, evaluated against general labels.
    """
    cleaned = _preprocess_all(docs)
    fold_results: list[tuple[float, float, float]] = []

    for i, doc in enumerate(docs):
        y_test = np.array(doc['labels'])
        if (y_test == 1).sum() == 0:
            continue

        train_sents: list[str] = [
            s for j, sents in enumerate(cleaned) if j != i for s in sents
        ]

        vec = make_vectorizer()
        X_train = vec.fit_transform(train_sents)
        X_test = vec.transform(cleaned[i])

        combined_pred = np.full(len(cleaned[i]), -1, dtype=int)

        for cat in CATEGORIES:
            cat_train = np.array([
                lbl for j, d in enumerate(docs) if j != i for lbl in d['cat_labels'][cat]
            ])

            if (cat_train == 1).sum() == 0:
                continue  # no positive examples for this category in train split

            cat_pred = _fit_predict(model_name, X_train, cat_train, X_test)
            combined_pred = np.where(cat_pred == 1, 1, combined_pred)

        fold_results.append(doc_metrics(y_test, combined_pred))

    return fold_results
