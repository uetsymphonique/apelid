import os
from typing import Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger
from .cfm import CFMVelocityField


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
) -> pd.DataFrame:
    """
    Minimal CFM pipeline (mirrors DABEL style):
      - Encode train/test
      - Train CFM velocity field on linear paths
      - Generate samples by reverse Euler
      - Inverse to raw and concatenate with original train

    No post-filter, no dedup, no extras.
    """
    logger.info(f"[CFM] Augmenting {class_name}: {len(train_df)} -> {tau}")

    need = tau - len(train_df)
    if need <= 0:
        logger.info("[CFM] Already enough samples; skipping")
        return train_df

    # Encode
    enc_train = encode_with_preprocessor(pre, train_df)
    enc_test = encode_with_preprocessor(pre, test_df)
    feat_cols = [c for c in enc_train.columns if c != 'Label']

    # Prepare training arrays
    x_data = enc_train[feat_cols].astype(np.float32).values

    # Train velocity field (per-class; y is not used in minimal variant)
    vf = CFMVelocityField(input_dim=len(feat_cols), random_state=random_state)
    vf.fit(x_data, y_data=None, num_pairs=num_pairs, random_state=random_state)

    # Generate
    gen_enc = vf.generate(num_samples=need, num_steps=num_steps, random_state=random_state)
    gen_df_enc = pd.DataFrame(gen_enc, columns=feat_cols)
    class_id = pre.encoders['label'].transform([class_name])[0]
    gen_df_enc['Label'] = class_id

    # Inverse to raw space
    gen_raw = pre.inverse_transform(gen_df_enc)

    # Concatenate
    augmented = pd.concat([train_df, gen_raw], ignore_index=True)
    logger.info(f"[CFM] {class_name}: {len(train_df)} -> {len(augmented)}")

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

    return augmented


