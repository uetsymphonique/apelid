from typing import Any

from .art_classifier import AdversarialWrapper

from utils.logging import get_logger
logger = get_logger(__name__)


class XGBWrapper(AdversarialWrapper):
    """ART wrapper for XGBoost models via ART's XGBoostClassifier.

    Note: ART's XGBoostClassifier expects a sklearn-like XGBClassifier; if your
    training flow saves native Booster, consider constructing an XGBClassifier and
    loading the model with `load_model` for compatibility.
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import XGBoostClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ART is required for XGBWrapper") from exc
        # Try to enable GPU prediction if requested and supported
        try:
            if self.device and str(self.device).startswith("cuda"):
                print_device = "CUDA"
                # sklearn API
                if hasattr(self.model, "set_params"):
                    try:
                        # xgboost>=2.0 prefers device='cuda'
                        self.model.set_params(**{"device": "cuda"})
                    except Exception:
                        pass
                # native Booster API
                if hasattr(self.model, "set_param"):
                    try:
                        self.model.set_param({"device": "cuda"})
                    except Exception:
                        pass
                
                logger.info("Device: CUDA (requested), attempting device=cuda")
            else:
                logger.info("Device: CPU")
        except Exception:
            # Best-effort; ignore if not applicable
            pass

        return XGBoostClassifier(model=self.model)


