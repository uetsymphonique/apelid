import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.logging import get_logger

from .art_classifier import AdversarialWrapper


logger = get_logger(__name__)


class DNNClassifier(AdversarialWrapper):
    """ART wrapper for the DNN model (PyTorch) trained under src/training/dnn.py.

    Usage:
      - Construct from an existing DNNModel instance
      - Or use `from_checkpoint` to load state_dict from .pth file
    """

    def build_estimator(self) -> Any:  # type: ignore[override]
        try:
            from art.estimators.classification import PyTorchClassifier
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch and ART are required for DNNClassifier") from exc

        # Underlying torch module
        torch_model = getattr(self.model, "model", None) or self.model
        loss = getattr(self.model, "criterion", None) or nn.CrossEntropyLoss()
        optimizer = getattr(self.model, "optimizer", None)

        device_type = "gpu" if (self.device and str(self.device).startswith("cuda")) else "cpu"
        # Log device status
        try:
            cuda_avail = bool(torch.cuda.is_available())
            num_devices = torch.cuda.device_count() if cuda_avail else 0
            logger.info(f"Device: {device_type.upper()} (CUDA available={cuda_avail}, devices={num_devices})")
        except Exception:
            logger.info(f"Device: {device_type.upper()}")

        estimator = PyTorchClassifier(
            model=torch_model,
            loss=loss,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            device_type=device_type,
        )
        return estimator

    # ---------- Factory helpers ----------
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        *,
        input_dim: int,
        num_classes: int,
        device: Optional[str] = None,
        dnn_hparams: Optional[Dict[str, Any]] = None,
    ) -> "DNNClassifier":
        """Create a DNNClassifier by loading weights from a .pth checkpoint.

        Parameters
        - ckpt_path: path to .pth state_dict saved by training
        - input_dim: number of input features for the DNN
        - num_classes: number of classes
        - device: 'cpu' | 'cuda' | 'auto' | None
        - dnn_hparams: optional hyperparams for constructing DNNModel
        """
        from training.dnn import DNNModel

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        hp = dnn_hparams or {}

        model = DNNModel(
            input_dim=int(input_dim),
            num_class=int(num_classes),
            lr=float(hp.get("lr", 1e-3)),
            weight_decay=float(hp.get("weight_decay", 1e-4)),
            batch_size=int(hp.get("batch_size", 512)),
            max_epochs=int(hp.get("epochs", 1)),  # not used for inference
            patience=int(hp.get("patience", 8)),
            device=None if (device == "auto") else device,
            random_state=int(hp.get("random_state", 42)),
        )

        # Load weights (state_dict) directly into underlying torch module
        map_location = "cpu" if device == "cpu" else None
        state_dict = torch.load(ckpt_path, map_location=map_location)
        # Accept both raw nn.Module or wrapped DNNModel with .model
        torch_module = getattr(model, "model", None) or model
        torch_module.load_state_dict(state_dict)
        model._is_fitted = True

        wrapper = cls(
            model=model,
            num_classes=num_classes,
            input_shape=(int(input_dim),),
            device=device,
            params=hp,
        )
        logger.info(f"Loaded checkpoint from {ckpt_path} on device={device}")
        return wrapper


