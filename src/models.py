from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_NAMES = ['SVM', 'LogisticRegression', 'RandomForest', 'XGBoost']


def make_model(name: str, scale_pos_weight: float = 9.0):
    """
    Create a fresh model instance per CV fold to avoid state leakage.
    scale_pos_weight is only used by XGBoost; others use class_weight='balanced'.
    """
    if name == 'SVM':
        return LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
    if name == 'LogisticRegression':
        return LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'
        )
    if name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
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
