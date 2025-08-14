import numpy as np
from typing import Optional

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


class CFMVelocityField:
    """
    Minimal Conditional Flow Matching (CFM) velocity field approximator.

    - Path: linear interpolation between data x_data and base noise x_base
      x(t) = (1 - t) * x_data + t * x_base, t in [0, 1]
    - Target velocity: v*(x(t), t) = x_base - x_data
    - Model: Multi-output XGBoost regressor to predict v(x, t)

    This implementation mirrors the core idea used in DABEL's CFM scripts,
    without auxiliary filtering/deduplication logic.
    """

    def __init__(
        self,
        input_dim: int,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        base_regressor = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
        )
        self.model = MultiOutputRegressor(base_regressor)

    def _sample_base(self, num_samples: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.normal(loc=0.0, scale=1.0, size=(num_samples, self.input_dim)).astype(np.float32)

    def fit(
        self,
        x_data: np.ndarray,
        y_data: Optional[np.ndarray] = None,
        num_pairs: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """
        Train the velocity field on linear paths.

        - x_data: encoded real samples (shape: [N, D], values typically in [0,1])
        - y_data: optional labels (unused in this minimal per-class setup)
        - num_pairs: optional number of training pairs; defaults to len(x_data)
        """
        rng = np.random.RandomState(random_state)
        num_samples = x_data.shape[0]
        if num_pairs is None:
            num_pairs = num_samples

        # Sample indices to build training pairs
        indices = rng.choice(num_samples, size=num_pairs, replace=num_pairs > num_samples)
        x_real = x_data[indices]
        t = rng.uniform(low=0.0, high=1.0, size=(num_pairs, 1)).astype(np.float32)
        x_base = self._sample_base(num_pairs, rng)

        x_t = (1.0 - t) * x_real + t * x_base
        v_target = (x_base - x_real).astype(np.float32)

        # Features: [x_t, t]
        features = np.concatenate([x_t, t], axis=1)

        self.model.fit(features, v_target)

    def predict_velocity(self, x: np.ndarray, t_scalar: float) -> np.ndarray:
        t_col = np.full((x.shape[0], 1), t_scalar, dtype=np.float32)
        feats = np.concatenate([x.astype(np.float32), t_col], axis=1)
        return self.model.predict(feats).astype(np.float32)

    def generate(
        self,
        num_samples: int,
        num_steps: int = 200,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Reverse-time Euler integration using learned velocity field.
        Start from Gaussian base; step t: 1 → 0 with Δt = 1/num_steps.
        """
        rng = np.random.RandomState(random_state)
        x = self._sample_base(num_samples, rng)
        dt = 1.0 / float(num_steps)

        # Use t grid on [1, 0]; evaluate velocity at current t then step x ← x - dt * v(x,t)
        for k in range(num_steps, 0, -1):
            t_val = float(k) / float(num_steps)
            v = self.predict_velocity(x, t_val)
            x = x - dt * v

        return x.astype(np.float32)


