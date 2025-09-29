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
ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
ENCODED_TRAIN_DIR = os.path.join(ENCODED_DIR, 'train')
CLEAN_MERGED_DIR = cic2018.CLEAN_MERGED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER
BENIGN_SAFE = cic2018.get_label_name('Benign')
# Use Benign major compressed encoded train as critic input (default)
BENIGN_TRAIN_ENC_PATH = os.path.join(ENCODED_TRAIN_DIR, f"cic2018_{BENIGN_SAFE}_encoded_compressed.csv")


ACCEPT_RATE_MAP = {
    # Cover all current MINORITY_LABELS with sensible defaults
    # 'Infilteration': 0.1,
    # 'SSH-Bruteforce': 0.45,
    'DoS attacks-GoldenEye': 0.40,
    'DoS attacks-Hulk': 0.35,
    'DoS attacks-Slowloris': 0.30,
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


def _load_clean_merged_opposite(label_name: str, subset: str = 'train'):
    try:
        safe = cic2018.get_label_name(label_name)
        base_dir = os.path.join(CLEAN_MERGED_DIR, subset)
        if label_name == 'Benign':
            # Prefer compressed if available
            cmc = os.path.join(base_dir, f"cic2018_{safe}_{subset}_clean_merged_compressed.csv")
            if os.path.exists(cmc):
                return pd.read_csv(cmc, low_memory=False)
        # Fallback (or non-Benign): original clean_merged
        cm = os.path.join(base_dir, f"cic2018_{safe}_{subset}_clean_merged.csv")
        if not os.path.exists(cm):
            raise FileNotFoundError(f"Opposite clean_merged not found: {cm}")
        return pd.read_csv(cm, low_memory=False)
    except Exception as e:
        logger.warning(f"[WGAN] Opposite clean_merged loader failed: {e}")
        return None


def _encode_like_minor(pre: CIC2018Preprocessor, df: pd.DataFrame, num_encoder: str) -> pd.DataFrame:
    df2 = pre.select_features_and_label(df.copy())
    logger.info(f"[+] Encoding like minority with {num_encoder} encoder")
    if num_encoder == 'quantile_normal':
        enc_df = pre.preprocess_encode_numerical_features_quantile(df2)
    elif num_encoder == 'standard':
        enc_df = pre.preprocess_encode_numerical_features_standard(df2)
    elif num_encoder == 'quantile_uniform':
        enc_df = pre.preprocess_encode_numerical_features_quantile_uniform(df2)
    elif num_encoder == 'minmax':
        enc_df = pre.preprocess_encode_numerical_features(df2)
    else:
        raise ValueError(f"Invalid numerical encoder: {num_encoder}")
    enc_df = pre.preprocess_encode_binary_features(enc_df)
    enc_df = pre.preprocess_encode_label(enc_df)
    enc_df = pre.preprocess_encode_categorical_features(enc_df)
    return enc_df


def run_wgan(pre: CIC2018Preprocessor, class_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tau: int, *, opposite_label: str = 'Benign', critic_id: int | None = None, num_encoder: str = 'quantile_uniform') -> pd.DataFrame:
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
    # Build opposite loader from clean_merged -> sample 4x -> encode
    def _sampled_opposite_loader():
        df_cm = _load_clean_merged_opposite(opposite_label, subset='train')
        try:
            if df_cm is not None and len(df_cm) > 0:
                need = max(len(train_df) * 4, 1)
                df_s = df_cm.sample(n=need, replace=True, random_state=1)
                return _encode_like_minor(pre, df_s, num_encoder=num_encoder)
        except Exception as e:
            logger.warning(f"[WGAN] Opposite sampling/encoding failed: {e}")
        # Fallback: if anything fails, try to at least encode whatever loaded
        if df_cm is not None and len(df_cm) > 0:
            try:
                return _encode_like_minor(pre, df_cm, num_encoder=num_encoder)
            except Exception:
                pass
        return None

    return wgan_generate(
        pre=pre,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        benign_loader=_sampled_opposite_loader,
        tau=tau,
        device='auto',
        accept_rate=ar,
        min_precision=0.95,
        options=opts,
        critic_id=critic_id,
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
    parser = argparse.ArgumentParser(description="CIC minority augmenting (WGAN/CFM/FDM)")
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan', 'cfm', 'fdm'],
                        help='Augmenting strategy to use')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Augment all predefined minority labels or provided labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MINORITY_LABELS,
                        help='Label names to augment when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=cic2018.MINORITY_LABELS,
                        help='Labels to exclude from augmentation')
    parser.add_argument('--opposite-label', type=str, default='Benign',
                        help='Label to use as critic opposite class (default Benign)')
    parser.add_argument('--num-encoder', '-n', type=str, default='quantile_uniform', choices=['minmax', 'quantile_uniform', 'standard', 'quantile_normal'],
                        help='Numerical encoder for opposite sampling encode (match minority encoding)')
    parser.add_argument('--tau', '-t', type=int, default=65800, help='Base target samples per class after augmentation')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)

    # We work on encoded train per label (no PCA/UMAP). Ensure directory exists.
    if not os.path.isdir(ENCODED_TRAIN_DIR):
        raise SystemExit(f"Encoded train directory not found: {ENCODED_TRAIN_DIR}. Run cli.cic2018.minor.encode --subset train first.")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Fit encoders on TRAIN first: python -m cli.cic2018.preprocessing.setup_encoders")

    
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        minority_labels = args.labels
    else:
        minority_labels = cic2018.MINORITY_LABELS
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        minority_labels = [lbl for lbl in minority_labels if lbl not in excludes]

    logger.info(f"[+] Minority labels to augment: {minority_labels}")

    tau_effective = args.tau
    # Resolve critic (opposite) label id for binary mapping
    try:
        critic_id = pre.encoders['label'].transform([args.opposite_label])[0]
    except Exception:
        critic_id = None
    for lbl in minority_labels:
        safe = cic2018.get_label_name(lbl)
        enc_train_path = os.path.join(ENCODED_TRAIN_DIR, f"cic2018_{safe}_encoded.csv")
        if not os.path.exists(enc_train_path):
            logger.warning(f"[+] Encoded train not found for {lbl}: {enc_train_path}; skip")
            continue
        train_df = pd.read_csv(enc_train_path, low_memory=False)
        
        # Load encoded test if available (for dedup keys in WGAN)
        # WGAN pipeline expects valid test_df for dedup, so provide at minimum a small sample
        encoded_test_dir = os.path.join(ENCODED_DIR, 'test')
        enc_test_path = os.path.join(encoded_test_dir, f"cic2018_{safe}_encoded.csv")
        if os.path.exists(enc_test_path):
            test_df = pd.read_csv(enc_test_path, low_memory=False)
            logger.info(f"[+] Loaded encoded test for dedup: {len(test_df)} rows")
        else:
            # Create minimal test_df by sampling from train_df to avoid WGAN pipeline issues
            test_df = train_df.sample(n=min(100, len(train_df)), random_state=42).copy()
            logger.info(f"[+] No encoded test found, using train sample as test_df for {lbl} ({len(test_df)} rows)")

        if args.augmenting_strategy == 'wgan':
            aug_train = run_wgan(pre, lbl, train_df, test_df, tau=tau_effective, opposite_label=args.opposite_label, critic_id=critic_id, num_encoder=args.num_encoder)
        elif args.augmenting_strategy == 'cfm':
            logger.error(f"[+] CFM augmentation not implemented yet for {lbl}")
            continue
        else:  # fdm
            logger.error(f"[+] FDM augmentation not implemented yet for {lbl}")
            continue

        # Save per-label outputs (encoded train only)
        enc_out = os.path.join(ENCODED_TRAIN_DIR, f"cic2018_{safe}_minority_{args.augmenting_strategy}_train_augmented_encoded.csv")
        os.makedirs(ENCODED_TRAIN_DIR, exist_ok=True)
        aug_train.to_csv(enc_out, index=False)
        logger.info(f"[+] Saved augmented minority train ENCODED ({lbl}): {enc_out} ({len(aug_train)})")

    logger.info("[+] CIC minority augmenting completed")


if __name__ == "__main__":
    main()


