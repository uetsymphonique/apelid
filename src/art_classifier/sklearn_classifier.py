from typing import Any

from .art_classifier import AdversarialWrapper


class SkleanWrapper(AdversarialWrapper):
    """ART wrapper for sklearn-compatible classifiers using ART's SklearnClassifier."""

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import SklearnClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ART is required for SkleanWrapper") from exc

        return SklearnClassifier(model=self.model)


