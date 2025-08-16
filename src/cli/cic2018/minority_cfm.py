import os
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from sklearn.model_selection import train_test_split
from resampling.data_augmentation.conditional_flow_matching.pipeline import (
    generate_augmented_samples_cfm,
)


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
RAW_PATH = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")


def split_per_class(df: pd.DataFrame, label_col: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    return (pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1))


def main():
    if not os.path.exists(RAW_PATH):
        raise SystemExit(f"Raw processed file not found: {RAW_PATH}. Run cli/cic2018/preprocessing first.")

    raw_df = pd.read_csv(RAW_PATH)
    logger.info(f"[CFM] Loaded raw: {len(raw_df)} rows")

    # Initialize preprocessor and load encoders
    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        pre.setup_encoders(raw_df)
        pre.save_encoders()

    # Fixed minority labels (<= 20k)
    minority_labels = [
        'DoS attacks-Slowloris',
        'DDOS attack-LOIC-UDP',
        'Brute Force -Web',
        'Brute Force -XSS',
        'SQL Injection',
        'DoS attacks-SlowHTTPTest',
        'FTP-BruteForce',
    ]
    logger.info(f"[CFM] Minority labels: {minority_labels}")

    augmented_frames = []
    test_frames = []

    for lbl in minority_labels:
        cls_df = raw_df[raw_df['Label'] == lbl].copy()
        if len(cls_df) == 0:
            logger.warning(f"[CFM] No samples for {lbl}; skip")
            continue
        train_df, test_df = split_per_class(cls_df, pre.label_column, test_size=0.3, random_state=42)
        aug_train = generate_augmented_samples_cfm(
            pre=pre,
            class_name=lbl,
            train_df=train_df,
            test_df=test_df,
            tau=14000,
            random_state=42,
            num_pairs=None,
            num_steps=200,
        )
        augmented_frames.append(aug_train)
        test_frames.append(test_df)

    if augmented_frames:
        minority_train = pd.concat(augmented_frames, ignore_index=True)
        out_train = os.path.join(DATA_FOLDER, 'cic_minority_train_augmented_raw.csv')
        minority_train.to_csv(out_train, index=False)
        logger.info(f"[CFM] Saved augmented minority train: {out_train} ({len(minority_train)})")
    else:
        logger.warning("[CFM] No minority classes processed (nothing to augment)")

    if test_frames:
        minority_test = pd.concat(test_frames, ignore_index=True)
        out_test = os.path.join(DATA_FOLDER, 'cic_minority_test_raw.csv')
        minority_test.to_csv(out_test, index=False)
        logger.info(f"[CFM] Saved minority test: {out_test} ({len(minority_test)})")

    logger.info("[CFM] CIC minority CFM processing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIC minority CFM processing")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)
    main()



