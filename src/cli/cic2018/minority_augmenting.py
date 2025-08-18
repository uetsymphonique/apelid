import os
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from sklearn.model_selection import train_test_split
from configs import cic2018

# WGAN
from resampling.data_augmentation.augmented_wgan.pipeline import (
    generate_augmented_samples as wgan_generate,
    AugmentOptions as WGANOptions,
)

# CFM
from resampling.data_augmentation.conditional_flow_matching.pipeline import (
    generate_augmented_samples_cfm as cfm_generate,
)

# FDM
from resampling.data_augmentation.forest_diffusion.pipeline import (
    generate_augmented_samples_fdm as fdm_generate,
)


logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
RAW_PATH = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")
ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER
BENIGN_SAFE = cic2018.get_label_name('Benign')
BENIGN_TRAIN_ENC_PATH = os.path.join(ENCODED_DIR, f"cic2018_{BENIGN_SAFE}_majority_train_compressed_encoded.csv")


ACCEPT_RATE_MAP = {
    'DoS attacks-Slowloris': 0.30,
    'DDOS attack-LOIC-UDP': 0.40,
    'Brute Force -Web': 0.45,
    'Brute Force -XSS': 0.50,
    'SQL Injection': 0.50,
    'DoS attacks-SlowHTTPTest': 0.50,
    'FTP-BruteForce': 0.50,
}


def split_per_class(df: pd.DataFrame, label_col: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    return (pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1))


def _benign_loader(pre: CIC2018Preprocessor):
    try:
        if not os.path.exists(BENIGN_TRAIN_ENC_PATH):
            raise FileNotFoundError(f"Benign encoded train not found: {BENIGN_TRAIN_ENC_PATH}")
        df = pd.read_csv(BENIGN_TRAIN_ENC_PATH)
        # Ensure label id matches Benign encoder id (robustness)
        try:
            benign_id = pre.encoders['label'].transform(['Benign'])[0]
            df = df[df['Label'] == benign_id]
        except Exception:
            pass
        return df
    except Exception as e:
        logger.warning(f"[WGAN] Benign loader failed: {e}")
        return None


def run_wgan(pre: CIC2018Preprocessor, class_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tau: int) -> pd.DataFrame:
    ar = ACCEPT_RATE_MAP.get(class_name, 0.20)
    opts = WGANOptions(
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
        trim_to_need=True,
        use_final_fill=True,
    )
    return wgan_generate(
        pre=pre,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        benign_loader=lambda: _benign_loader(pre),
        tau=tau,
        device='auto',
        accept_rate=ar,
        min_precision=0.95,
        options=opts,
    )


def run_cfm(pre: CIC2018Preprocessor, class_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tau: int) -> pd.DataFrame:
    return cfm_generate(
        pre=pre,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        tau=tau,
        random_state=42,
        num_pairs=None,
        num_steps=200,
        n_t=50,
        duplicate_K=10,
        n_jobs=-1,
    )


def run_fdm(pre: CIC2018Preprocessor, class_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tau: int) -> pd.DataFrame:
    return fdm_generate(
        pre=pre,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        tau=tau,
        n_t=50,
        duplicate_K=100,
        n_jobs=-1,
        seed=42,
        batch_size=None,
    )


def main():
    all_minority_labels = [
        'DoS attacks-Slowloris',
        'DDOS attack-LOIC-UDP',
        'Brute Force -Web',
        'Brute Force -XSS',
        'SQL Injection',
        'DoS attacks-SlowHTTPTest',
        'FTP-BruteForce',
    ]

    parser = argparse.ArgumentParser(description="CIC minority augmenting (WGAN/CFM/FDM)")
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan', 'cfm', 'fdm'],
                        help='Augmenting strategy to use')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Augment all predefined minority labels or a single label')
    parser.add_argument('--label', type=str, default=None, choices=all_minority_labels,
                        help='Label name to augment when mode=label')
    parser.add_argument('--tau', '-t', type=int, default=14000, help='Base target samples per class after augmentation')
    parser.add_argument('--extra-train', type=int, default=2000, help='Fixed extra samples to add to augmented train (for later ENN)')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not os.path.exists(RAW_PATH):
        raise SystemExit(f"Raw processed file not found: {RAW_PATH}. Run cli/cic2018/preprocess_data first.")

    raw_df = pd.read_csv(RAW_PATH)
    logger.info(f"[+] Loaded raw: {len(raw_df)} rows")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        pre.setup_encoders(raw_df)
        pre.save_encoders()

    
    if args.mode == 'label':
        if not args.label:
            raise SystemExit("--label is required when --mode label")
        minority_labels = [args.label]
    else:
        minority_labels = all_minority_labels

    logger.info(f"[+] Minority labels to augment: {minority_labels}")

    os.makedirs(RAW_DIR, exist_ok=True)

    tau_effective = args.tau + args.extra_train
    for lbl in minority_labels:
        cls_df = raw_df[raw_df['Label'] == lbl].copy()
        if len(cls_df) == 0:
            logger.warning(f"[+] No samples for {lbl}; skip")
            continue
        train_df, test_df = split_per_class(cls_df, pre.label_column, test_size=0.3, random_state=42)

        if args.augmenting_strategy == 'wgan':
            aug_train = run_wgan(pre, lbl, train_df, test_df, tau=tau_effective)
        elif args.augmenting_strategy == 'cfm':
            aug_train = run_cfm(pre, lbl, train_df, test_df, tau=tau_effective)
        else:  # fdm
            aug_train = run_fdm(pre, lbl, train_df, test_df, tau=tau_effective)

        # Save per-label outputs
        safe = cic2018.get_label_name(lbl)
        # Augmented train is already ENCODED (pipelines return encoded)
        enc_out = os.path.join(ENCODED_DIR, f"cic2018_{safe}_minority_{args.augmenting_strategy}_train_augmented_encoded.csv")
        os.makedirs(ENCODED_DIR, exist_ok=True)
        aug_train.to_csv(enc_out, index=False)
        logger.info(f"[+] Saved augmented minority train ENCODED ({lbl}): {enc_out} ({len(aug_train)})")
        # Test split (raw)
        out_test = os.path.join(RAW_DIR, f"cic2018_{safe}_minority_test_raw.csv")
        test_df.to_csv(out_test, index=False)
        logger.info(f"[+] Saved minority test ({lbl}): {out_test} ({len(test_df)})")

    logger.info("[+] CIC minority augmenting completed")


if __name__ == "__main__":
    main()


