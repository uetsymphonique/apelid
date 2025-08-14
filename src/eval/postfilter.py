import os
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from utils.logging import get_logger

logger = get_logger(__name__)


class PostFilterClassifier:
    """
    Lightweight wrapper around a binary classifier used to post-filter
    generated samples. Prefers XGBoost if available; otherwise falls back to
    GradientBoostingClassifier.
    """

    def __init__(self):
        self.model = None
        self.feature_names: Optional[Sequence[str]] = None
        self.threshold: Optional[float] = None

    def _make_model(self, pos_weight: float = 1.0):
        try:
            from xgboost import XGBClassifier  # type: ignore
            logger.info("[PostFilter] Using XGBoost classifier")
            return XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=4,
                scale_pos_weight=pos_weight,
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.info("[PostFilter] Using GradientBoosting classifier (fallback)")
            return GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        # Compute class weight for imbalance
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        pos_weight = (neg / pos) if pos > 0 else 1.0
        self.model = self._make_model(pos_weight=pos_weight)
        self.model.fit(X.values, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PostFilterClassifier not trained")
        # Align features
        if self.feature_names is not None:
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                logger.warning(f"[PostFilter] Missing features in input: {missing}")
            X = X[[f for f in self.feature_names if f in X.columns]]
        proba = self.model.predict_proba(X.values)
        # Return probability of positive class
        return proba[:, 1]

    def calibrate_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray,
                            min_precision: float = 0.90) -> float:
        """
        Choose probability threshold to achieve at least min_precision and
        maximize recall (equivalently F1 among feasible points). Falls back to
        best F1 overall if constraint cannot be met. Stores and returns threshold.
        """
        scores = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, scores)
        # precision_recall_curve returns len(thresholds)+1 points; align by skipping the first point
        precision = precision[:-1]
        recall = recall[:-1]
        if thresholds.size == 0:
            self.threshold = 0.5
            logger.warning("[PostFilter] No thresholds from PR curve; defaulting to 0.5")
            return self.threshold

        # Feasible set: precision >= min_precision
        feasible_idx = np.where(precision >= min_precision)[0]
        if feasible_idx.size > 0:
            f1 = 2 * precision[feasible_idx] * recall[feasible_idx] / (
                np.maximum(precision[feasible_idx] + recall[feasible_idx], 1e-12)
            )
            best = feasible_idx[np.argmax(f1)]
            thr = float(thresholds[best])
            self.threshold = thr
            logger.info(
                f"[PostFilter] Calibrated threshold={thr:.4f} (min_precision={min_precision}, P={precision[best]:.3f}, R={recall[best]:.3f})"
            )
            return thr

        # Fallback: best F1 overall
        f1_all = 2 * precision * recall / (np.maximum(precision + recall, 1e-12))
        best_all = int(np.argmax(f1_all))
        thr = float(thresholds[best_all])
        self.threshold = thr
        logger.info(
            f"[PostFilter] Fallback threshold={thr:.4f} (best F1={f1_all[best_all]:.3f}, P={precision[best_all]:.3f}, R={recall[best_all]:.3f})"
        )
        return thr

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "threshold": self.threshold,
        }, path)
        logger.info(f"[PostFilter] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        data = joblib.load(path)
        obj.model = data["model"]
        obj.feature_names = data.get("features")
        obj.threshold = data.get("threshold")
        logger.info(f"[PostFilter] Loaded from {path}")
        return obj


