import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def encode_with_preprocessor(pre, df: pd.DataFrame) -> pd.DataFrame:
    df_enc = pre.preprocess_encode_numerical_features(df.copy())
    df_enc = pre.preprocess_encode_binary_features(df_enc)
    df_enc = pre.preprocess_encode_label(df_enc)
    df_enc = pre.preprocess_encode_categorical_features(df_enc)
    return df_enc


def generate_augmented_samples_fdm(
    pre,
    class_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tau: int = 14000,
    n_t: int = 50,
    duplicate_K: int = 100,
    n_jobs: int = -1,
    seed: int = 42,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Forest Diffusion Model pipeline (DABEL-aligned):
      - Encode train/test
      - Train ForestDiffusionModel on encoded features + labels
      - Generate encoded samples (Xy) in the encoded space
      - RETURN ENCODED augmented dataframe (no inverse)
    """
    need = tau - len(train_df)
    logger.info(f"[FDM] Augmenting {class_name}: {len(train_df)} -> {tau}")
    if need <= 0:
        logger.info("[FDM] Already enough samples; skipping")
        return train_df

    logger.info("[FDM] Encoding train/test with preprocessor …")
    enc_train = encode_with_preprocessor(pre, train_df)
    enc_test = encode_with_preprocessor(pre, test_df)
    feat_cols: Iterable[str] = [c for c in enc_train.columns if c != 'Label']
    logger.info(
        f"[FDM] Encoded shapes: train={enc_train.shape}, test={enc_test.shape}, features={len(list(feat_cols))}"
    )

    # Prepare numpy arrays for ForestDiffusionModel
    X_train = enc_train[feat_cols].astype(np.float32).values
    y_train = enc_train['Label'].astype(np.int32).values

    try:
        from ForestDiffusion import ForestDiffusionModel as ForestFlowModel
    except Exception as e:
        raise RuntimeError(
            f"ForestDiffusion package not available: {e}. Please install and retry."
        )

    logger.info(
        f"[FDM] Training ForestDiffusionModel (n_t={n_t}, duplicate_K={duplicate_K}, n_jobs={n_jobs}, seed={seed}) …"
    )
    fdm = ForestFlowModel(
        X_train,
        label_y=y_train,
        n_t=n_t,
        duplicate_K=duplicate_K,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="flow",
        n_jobs=n_jobs,
        seed=seed,
    )

    # Determine request size
    request = batch_size if batch_size is not None else max(need, 2000)
    logger.info(f"[FDM] Generating encoded samples: request={request}")
    Xy_gen = fdm.generate(batch_size=request)

    # Build encoded DataFrame matching enc_train columns
    gen_df_enc = pd.DataFrame(Xy_gen, columns=list(feat_cols) + ['Label'])
    # Keep only needed amount
    if len(gen_df_enc) > need:
        gen_df_enc = gen_df_enc.sample(n=need, random_state=seed)
    # Force label to the target class and ensure integer dtype for inverse_transform
    class_id = pre.encoders['label'].transform([class_name])[0]
    gen_df_enc['Label'] = int(class_id)
    logger.info(f"[FDM] Generated encoded: {len(gen_df_enc)} rows (labels set to class_id={class_id})")

    # Return ENCODED augmented dataframe (enc_train + gen_df_enc)
    augmented_enc = pd.concat([enc_train, gen_df_enc], ignore_index=True)
    logger.info(f"[FDM] {class_name} augmented (encoded): {len(enc_train)} + {len(gen_df_enc)} → {len(augmented_enc)}")
    return augmented_enc


