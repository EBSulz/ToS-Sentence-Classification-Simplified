from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_NAMES = ['SVM', 'LogisticRegression', 'XGBoost']


def make_model(name: str, scale_pos_weight: float = 9.0):
    """
    Create a fresh model instance.

    scale_pos_weight is only used by XGBoost; SVM/LR use class_weight='balanced'.
    A fresh instance is needed each LOO fold to avoid state leakage.
    """
    if name == 'SVM':
        return LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
    if name == 'LogisticRegression':
        return LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'
        )
    if name == 'XGBoost':
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            verbosity=0,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")
