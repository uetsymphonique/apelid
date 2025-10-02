from __future__ import annotations

from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.logging import get_logger

from .model import Model, log_gpu_status

logger = get_logger(__name__)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, num_class: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNModel(Model):
    def __init__(
        self,
        *,
        input_dim: int,
        num_class: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 8,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.input_dim = int(input_dim)
        self.num_class = int(num_class)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Check and log GPU status
        log_gpu_status(str(self.device), "DNN")
        
        torch.manual_seed(self.random_state or 42)
        self.model = _MLP(self.input_dim, self.num_class).to(self.device)
        self._best_state: Optional[dict[str, Any]] = None

    def _make_loader(self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool) -> DataLoader:
        X_t = torch.from_numpy(X).float()
        if y is None:
            ds = TensorDataset(X_t)
        else:
            y_t = torch.from_numpy(y).long()
            ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None and y_val is not None else None
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float('inf')
        wait = 0
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_n = 0
            for batch in train_loader:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
                bs = yb.size(0)
                train_loss += float(loss.item()) * bs
                train_n += bs
            
            avg_train_loss = train_loss / max(1, train_n)

            if val_loader is None:
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {avg_train_loss:.6f}")
                continue
                
            # Validation phase
            self.model.eval()
            total_loss = 0.0
            total_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    bs = yb.size(0)
                    total_loss += float(loss.item()) * bs
                    total_n += bs
            avg_val = total_loss / max(1, total_n)
            
            if avg_val < best_val - 1e-6:
                best_val = avg_val
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
                improved = "improved"
            else:
                wait += 1
                improved = ""
                if wait >= self.patience:
                    logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val:.6f} (Best: {best_val:.6f}, Wait: {wait}/{self.patience}) {improved}")
                    logger.debug(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Log every epoch or every 10 epochs if early stopping
            if wait < self.patience and (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.max_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val:.6f} (Best: {best_val:.6f}, Wait: {wait}/{self.patience}) {improved}")

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        loader = self._make_loader(X, None, shuffle=False)
        preds: list[int] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().numpy().tolist())
        return np.asarray(preds, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        loader = self._make_loader(X, None, shuffle=False)
        probas: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                prob = torch.softmax(logits, dim=1)
                probas.append(prob.cpu().numpy())
        return np.concatenate(probas, axis=0) if probas else None

    def save_model(self, path: str) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load_model(cls, path: str) -> "DNNModel":
        raise NotImplementedError("Please initialize DNNModel with proper input_dim/num_class and call load_state_dict externally.")


