from typing import Any

from .art_classifier import AdversarialWrapper

from utils.logging import get_logger

logger = get_logger(__name__)
class CatBoostWrapper(AdversarialWrapper):
    """ART wrapper for CatBoostClassifier models.

    Expects a fitted catboost.CatBoostClassifier or its sklearn-compatible wrapper.
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import CatBoostARTClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ART is required for CatBoostWrapper") from exc
        # Try to log/enable GPU intent
        if self.device and str(self.device).startswith("cuda"):
            logger.info("Device: CUDA (requested). Inference GPU usage depends on CatBoost build; proceeding.")
        else:
            logger.info("Device: CPU")

        return CatBoostARTClassifier(model=self.model)


