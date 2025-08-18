import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from utils.logging import get_logger
from .cfm import CFMVelocityField
from joblib import Parallel, delayed

try:
    import torch
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
    _HAS_TORCHCFM = True
except Exception:
    _HAS_TORCHCFM = False
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


logger = get_logger(__name__)


def encode_with_preprocessor(pre, df: pd.DataFrame) -> pd.DataFrame:
    df_enc = pre.preprocess_encode_numerical_features(df.copy())
    df_enc = pre.preprocess_encode_binary_features(df_enc)
    df_enc = pre.preprocess_encode_label(df_enc)
    df_enc = pre.preprocess_encode_categorical_features(df_enc)
    return df_enc


def generate_augmented_samples_cfm(
    pre,
    class_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tau: int = 14000,
    random_state: int = 42,
    num_pairs: Optional[int] = None,
    num_steps: int = 200,
    n_t: int = 50,
    duplicate_K: int = 10,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Default CFM pipeline (DABEL-style when torchcfm + xgboost are available):
      - Encode train/test
      - If torchcfm + xgboost available: build (x_t, u_t) using ConditionalFlowMatcher,
        train per-(t, class, feature) XGB regressors, integrate Euler to generate
        encoded samples with class prior
      - Else: fallback to minimal velocity-field approximation
      - RETURN ENCODED augmented dataframe (no inverse)
    """
    logger.info(f"[CFM] Augmenting {class_name}: {len(train_df)} -> {tau}")

    need = tau - len(train_df)
    if need <= 0:
        logger.info("[CFM] Already enough samples; skipping")
        return train_df

    # Encode
    logger.info("[CFM] Encoding train/test with preprocessor …")
    enc_train = encode_with_preprocessor(pre, train_df)
    enc_test = encode_with_preprocessor(pre, test_df)
    feat_cols: Iterable[str] = [c for c in enc_train.columns if c != 'Label']
    logger.info(f"[CFM] Encoded shapes: train={enc_train.shape}, test={enc_test.shape}, features={len(list(feat_cols))}")

    if _HAS_TORCHCFM and _HAS_XGB:
        logger.info("[CFM] Using DABEL-style CFM (torchcfm + XGBoost) …")
        rng = np.random.RandomState(random_state)
        X_real = enc_train[feat_cols].astype(np.float32).values
        y_ids = enc_train['Label'].values
        b, c = X_real.shape

        # Class prior from training class
        classes, counts = np.unique(y_ids, return_counts=True)
        y_probs = counts / counts.sum()
        logger.info(f"[CFM] Classes={len(classes)}, prior={dict(zip(map(int, classes), map(int, counts)))}")

        # Duplicate and build base
        X1 = np.tile(X_real, (duplicate_K, 1)).astype(np.float32)
        X0 = rng.normal(loc=0.0, scale=1.0, size=X1.shape).astype(np.float32)
        logger.info(f"[CFM] Duplicate_K={duplicate_K}, X1/X0 shape={X1.shape}")

        # Build masks per class on duplicated axis
        mask_y = {}
        for cls in classes:
            base_mask = (y_ids == cls)
            mask_y[int(cls)] = np.tile(base_mask, duplicate_K)

        # Time grid and CFM
        FM = ConditionalFlowMatcher(sigma=0.0)
        t_levels = np.linspace(1e-3, 1.0, num=n_t)
        X_train_t = np.zeros((n_t, X0.shape[0], X0.shape[1]), dtype=np.float32)
        U_train_t = np.zeros((n_t, X0.shape[0], X0.shape[1]), dtype=np.float32)
        logger.info(f"[CFM] Building (x_t, u_t) over n_t={n_t} time steps …")
        for i, tval in enumerate(t_levels):
            t_vec = torch.ones(X0.shape[0]) * float(tval)
            _, xt, ut = FM.sample_location_and_conditional_flow(
                torch.from_numpy(X0), torch.from_numpy(X1), t=t_vec
            )
            X_train_t[i] = xt.numpy().astype(np.float32)
            U_train_t[i] = ut.numpy().astype(np.float32)
        logger.info("[CFM] Built training tensors for all time steps")

        # Train XGB regressors per (time, class, feature)
        def _fit_regressor(X_in: np.ndarray, y_in: np.ndarray):
            model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method='hist',
                n_jobs=1,
                random_state=random_state,
            )
            non_nan = ~np.isnan(y_in)
            model.fit(X_in[non_nan, :], y_in[non_nan])
            return model

        tasks = []
        for i in range(n_t):
            for cls in classes:
                cls_mask = mask_y[int(cls)]
                Xt_cls = X_train_t[i][cls_mask, :]
                Ut_cls = U_train_t[i][cls_mask, :]
                for k in range(c):
                    tasks.append((i, int(cls), k, Xt_cls, Ut_cls[:, k]))
        total_models = len(tasks)
        logger.info(f"[CFM] Training regressors: total={total_models} (n_t={n_t} × classes={len(classes)} × dims={c}), n_jobs={n_jobs}")
        fitted = Parallel(n_jobs=n_jobs, backend='threading', max_nbytes=None)(
            delayed(_fit_regressor)(Xt, uk) for (_, _, _, Xt, uk) in tasks
        )
        # Reassemble into regr[cls][i][k]
        regr = {int(cls): [ [None]*c for _ in range(n_t) ] for cls in classes}
        idx = 0
        for (i, cls, k, _, _ ) in tasks:
            regr[cls][i][k] = fitted[idx]
            idx += 1
        logger.info("[CFM] Finished training XGB regressors")

        # Define velocity model and Euler solver
        def _predict_velocity(t_scalar: float, xt_flat: np.ndarray, label_mask: dict[int, np.ndarray]) -> np.ndarray:
            xt = xt_flat.reshape(-1, c)
            out = np.zeros_like(xt)
            ti = int(round(t_scalar * (n_t - 1)))
            for cls in classes:
                m = label_mask[int(cls)]
                if not np.any(m):
                    continue
                Xt_sub = xt[m, :]
                preds = np.zeros_like(Xt_sub)
                for k in range(c):
                    preds[:, k] = regr[int(cls)][ti][k].predict(Xt_sub)
                out[m, :] = preds
            return out.reshape(-1)

        def _euler_solve(x0_vec: np.ndarray, steps: int, vfunc, label_mask):
            h = 1.0 / (steps - 1)
            x = x0_vec
            t = 0.0
            for _ in range(steps - 1):
                x = x + h * vfunc(t, x, label_mask)
                t += h
            return x

        # Generate encoded samples
        need = tau - len(train_df)
        if need <= 0:
            logger.info("[CFM] Already enough samples; skipping")
            return train_df
        logger.info(f"[CFM] Generating encoded samples: need={need}, c={c}, steps={n_t}")
        x0 = rng.normal(size=(need, c)).astype(np.float32)
        # Sample labels per class prior
        label_y_fake = rng.choice(classes, size=need, p=y_probs)
        mask_y_fake = {int(cls): (label_y_fake == cls) for cls in classes}
        ode = _euler_solve(x0.reshape(-1), steps=n_t, vfunc=_predict_velocity, label_mask=mask_y_fake)
        sol = ode.reshape(need, c)
        gen_df_enc = pd.DataFrame(sol, columns=list(feat_cols))
        gen_df_enc['Label'] = label_y_fake.astype(int)
        logger.info(f"[CFM] Generated encoded: {len(gen_df_enc)} rows")
    else:
        # Fallback to minimal CFMVelocityField
        logger.info("[CFM] Fallback to minimal velocity-field variant …")
        x_data = enc_train[feat_cols].astype(np.float32).values
        vf = CFMVelocityField(input_dim=len(feat_cols), random_state=random_state)
        vf.fit(x_data, y_data=None, num_pairs=num_pairs, random_state=random_state)
        gen_enc = vf.generate(num_samples=need, num_steps=num_steps, random_state=random_state)
        gen_df_enc = pd.DataFrame(gen_enc, columns=feat_cols)
        class_id = pre.encoders['label'].transform([class_name])[0]
        gen_df_enc['Label'] = class_id
        logger.info(f"[CFM] Generated encoded (fallback): {len(gen_df_enc)} rows")

    # Return ENCODED augmented dataframe (enc_train + gen_df_enc)
    augmented_enc = pd.concat([enc_train, gen_df_enc], ignore_index=True)
    logger.info(f"[CFM] {class_name} augmented (encoded): {len(enc_train)} + {len(gen_df_enc)} → {len(augmented_enc)}")

    # Save basic artifacts (optional but useful)
    safe = class_name.lower().replace(' ', '_').replace('/', '_')
    os.makedirs('models', exist_ok=True)
    # Export a small sample of generated encoded for quick inspection
    try:
        sample_path = f"models/cfm_samples_{safe}.csv"
        pd.concat([enc_train[feat_cols].head(100), gen_df_enc[feat_cols].head(100)],
                  keys=['real_head', 'gen_head']).to_csv(sample_path)
        logger.info(f"[CFM] Wrote sample comparison: {sample_path}")
    except Exception:
        pass

    return augmented_enc


