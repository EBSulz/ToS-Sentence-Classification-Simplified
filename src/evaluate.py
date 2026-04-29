import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def doc_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for the positive class (1 = unfair)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    return float(p), float(r), float(f1)


def macro_average(
    fold_results: list[tuple[float, float, float]]
) -> tuple[float, float, float]:
    """Macro-average (P, R, F1) tuples over folds, matching paper methodology."""
    ps, rs, f1s = zip(*fold_results)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))
