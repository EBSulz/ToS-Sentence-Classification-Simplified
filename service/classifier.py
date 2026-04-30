"""
C2Classifier: 8 LinearSVC models (one per unfairness category), OR-combined.
Trained on the full corpus; cached to disk with joblib.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import CATEGORIES
from src.preprocessing import clean
from src.features import make_vectorizer
from src.models import make_model

logger = logging.getLogger(__name__)

_CATEGORY_LABELS = {
    'a': 'Arbitration',
    'ch': 'Unilateral Change',
    'cr': 'Content Removal',
    'j': 'Jurisdiction',
    'law': 'Choice of Law',
    'ltd': 'Limitation of Liability',
    'ter': 'Unilateral Termination',
    'use': 'Contract by Using',
}

_VEC_FILE = "c2_vectorizer.joblib"
_MOD_FILE = "c2_models.joblib"


class C2Classifier:
    def __init__(self) -> None:
        self._vec = None
        self._models: dict = {}  # cat -> fitted LinearSVC (or None if 0 positives)
        self._ready = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, docs: list[dict]) -> None:
        """Train on ALL docs (no CV). Called once at service startup."""
        logger.info("Training C2 classifier on %d documents...", len(docs))

        all_sentences = [clean(s) for d in docs for s in d['sentences']]
        self._vec = make_vectorizer()
        X = self._vec.fit_transform(all_sentences)

        offset = 0
        doc_slices = []
        for d in docs:
            n = len(d['sentences'])
            doc_slices.append(slice(offset, offset + n))
            offset += n

        self._models = {}
        for cat in CATEGORIES:
            y = np.array([lbl for d in docs for lbl in d['cat_labels'][cat]])
            pos = int((y == 1).sum())
            if pos == 0:
                logger.debug("Category %s: 0 positives, skipping", cat)
                self._models[cat] = None
                continue
            model = make_model('SVM', scale_pos_weight=(len(y) - pos) / pos)
            model.fit(X, y)
            self._models[cat] = model
            logger.debug("Category %s: trained on %d pos / %d total", cat, pos, len(y))

        self._ready = True
        logger.info("C2 classifier ready.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vec, cache_dir / _VEC_FILE)
        joblib.dump(self._models, cache_dir / _MOD_FILE)
        logger.info("C2 models saved to %s", cache_dir)

    def load(self, cache_dir: Path) -> None:
        self._vec = joblib.load(cache_dir / _VEC_FILE)
        self._models = joblib.load(cache_dir / _MOD_FILE)
        self._ready = True
        logger.info("C2 models loaded from %s", cache_dir)

    def load_or_train(self, docs: list[dict], cache_dir: Path) -> None:
        """Load cached models if available, otherwise train and save."""
        vec_path = cache_dir / _VEC_FILE
        mod_path = cache_dir / _MOD_FILE
        if vec_path.exists() and mod_path.exists():
            logger.info("Cache found — loading C2 models from %s", cache_dir)
            self.load(cache_dir)
        else:
            logger.info("No cache found — training C2 models...")
            self.train(docs)
            self.save(cache_dir)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, sentence: str) -> dict:
        """
        Returns a dict matching ClassifyResponse schema:
          {is_unfair: bool, categories: list[str], details: dict[str, bool]}
        """
        if not self._ready:
            raise RuntimeError("C2Classifier is not trained/loaded yet.")

        cleaned = clean(sentence)
        X = self._vec.transform([cleaned])

        details: dict[str, bool] = {}
        for cat in CATEGORIES:
            model = self._models.get(cat)
            if model is None:
                details[cat] = False
            else:
                pred = model.predict(X)[0]
                details[cat] = bool(pred == 1)

        fired = [cat for cat, v in details.items() if v]
        return {
            "is_unfair": len(fired) > 0,
            "categories": fired,
            "details": details,
        }
