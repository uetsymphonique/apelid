import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np



class AdversarialWrapper(abc.ABC):
    """Common interface for adversarial robustness wrappers using ART.

    This wrapper adapts a trained model to ART's estimator interface and exposes
    convenience methods to generate adversarial samples and evaluate robustness.
    """

    def __init__(
        self,
        *,
        model: Any,
        num_classes: int,
        input_shape: Tuple[int, ...],
        clip_values: Tuple[float, float],
        device: str | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.clip_values = tuple(clip_values)
        self.device = device or "cpu"
        self.params = params.copy() if params else {}
        self._estimator = None  # Underlying ART estimator

    @abc.abstractmethod
    def build_estimator(self) -> Any:
        """Create and return the ART estimator for the underlying model."""

    def get_estimator(self) -> Any:
        if self._estimator is None:
            self._estimator = self.build_estimator()
        return self._estimator

    # Basic API
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        estimator = self.get_estimator()
        # Ensure float32 to avoid dtype mismatch for torch-based estimators
        X_safe = X.astype(np.float32, copy=False)
        return estimator.predict(X_safe)  # ART returns probabilities for classifiers

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        accuracy = float((y_pred == y_true).mean())
        return {"accuracy": accuracy}

    # Params API
    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)





