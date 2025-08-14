import os
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from sklearn.model_selection import train_test_split
from resampling.data_augmentation.augmented_wgan.pipeline import (
    generate_augmented_samples,
    AugmentOptions,
)


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
RAW_PATH = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")
ENCODED_MAJ_TRAIN_PATH = os.path.join(DATA_FOLDER, "cic_majority_train_compressed_encoded.csv")

# Per-class accept_rate for top-k selection (based on class sizes)
ACCEPT_RATE_MAP = {
    'DoS attacks-Slowloris': 0.30,    # ~9.9k
    'DDOS attack-LOIC-UDP': 0.40,     # ~1.7k
    'Brute Force -Web': 0.45,         # 555
    'Brute Force -XSS': 0.50,         # 228
    'SQL Injection': 0.50,            # 84
    'DoS attacks-SlowHTTPTest': 0.50, # 55
    'FTP-BruteForce': 0.50,           # 53
}


def find_minority_labels(df: pd.DataFrame, tau_major: int = 20000):
    counts = df['Label'].value_counts()
    minority_labels = counts[counts <= tau_major].index.tolist()
    # Ensure we don't include empty labels
    minority_labels = [lbl for lbl in minority_labels if counts[lbl] > 0]
    return minority_labels, counts.to_dict()


def split_per_class(df: pd.DataFrame, label_col: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    return (pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1))


def _build_encoded_keys(df_enc: pd.DataFrame, feat_cols: list[str], decimals: int = 6) -> set:
    arr = df_enc[feat_cols].astype(float).round(decimals).values
    return set(map(tuple, arr))


def augment_minority_class(class_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           pre: CIC2018Preprocessor, tau: int = 14000, device: str = 'auto') -> pd.DataFrame:
    ar = ACCEPT_RATE_MAP.get(class_name, 0.20)
    
    # benign_loader for CIC reads encoded majority train and filters Benign
    def benign_loader():
        try:
            df = pd.read_csv(ENCODED_MAJ_TRAIN_PATH)
            benign_id = pre.encoders['label'].transform(['Benign'])[0]
            return df[df['Label'] == benign_id]
        except Exception as e:
            logger.warning(f"[-] Benign loader failed: {e}")
            return None
    
    opts = AugmentOptions(
        use_benign_for_critic=True,
        critic_epochs=60,
        wgan_iterations=10000,
        d_iter=5,
        use_gp=True,
        accept_rate=ar,
        request_multiplier=3.0,
        max_rounds=40,
        use_postfilter=True,
        min_precision=0.95,
        use_encoded_dedup=True,
        use_raw_dedup=True,
        trim_to_need=True,
        use_final_fill=True,
    )
    
    return generate_augmented_samples(
        pre=pre,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        benign_loader=benign_loader,
        tau=tau,
        device=device,
        accept_rate=ar,
        min_precision=0.95,
        options=opts,
    )


def main():
    # setup_logging("DEBUG")

    if not os.path.exists(RAW_PATH):
        raise SystemExit(f"Raw processed file not found: {RAW_PATH}. Run cli-cic.preprocessing first.")

    raw_df = pd.read_csv(RAW_PATH)
    logger.info(f"[+] Loaded raw: {len(raw_df)} rows")

    # Initialize preprocessor and load encoders
    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        pre.setup_encoders(raw_df)
        pre.save_encoders()

    # Determine minority labels (<= 20k)
    # minority_labels, counts = find_minority_labels(raw_df, tau_major=20000)
    minority_labels = ['DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP', 
                       'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 
                       'DoS attacks-SlowHTTPTest', 'FTP-BruteForce']
    logger.info(f"[+] Minority labels (<= 20000): {minority_labels}")

    augmented_frames = []
    test_frames = []

    for lbl in minority_labels:
        cls_df = raw_df[raw_df['Label'] == lbl].copy()
        if len(cls_df) == 0:
            continue
        train_df, test_df = split_per_class(cls_df, pre.label_column, test_size=0.3, random_state=42)
        aug_train = augment_minority_class(lbl, train_df, test_df, pre, tau=14000, device='auto')
        augmented_frames.append(aug_train)
        test_frames.append(test_df)

    if augmented_frames:
        minority_train = pd.concat(augmented_frames, ignore_index=True)
        out_train = os.path.join(DATA_FOLDER, 'cic_minority_train_augmented_raw.csv')
        minority_train.to_csv(out_train, index=False)
        logger.info(f"[+] Saved augmented minority train: {out_train} ({len(minority_train)})")
    else:
        logger.warning("[-] No minority classes processed (nothing to augment)")

    if test_frames:
        minority_test = pd.concat(test_frames, ignore_index=True)
        out_test = os.path.join(DATA_FOLDER, 'cic_minority_test_raw.csv')
        minority_test.to_csv(out_test, index=False)
        logger.info(f"[+] Saved minority test: {out_test} ({len(minority_test)})")

    logger.info("[+] CIC minority WGAN processing completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIC minority WGAN processing")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)
    main()


