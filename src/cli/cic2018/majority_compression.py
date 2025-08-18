import os
import argparse
import numpy as np
import pandas as pd
from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from resampling.undersampling.kmeans import KMeansCompressor
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
ENCODED_PATH = os.path.join(DATA_FOLDER, "CIC2018_encoded.csv")
ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


def split_train_test(df, test_size=0.3, random_state=42):
    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


def _load_encoded_union(pre: CIC2018Preprocessor, labels: list[str]) -> pd.DataFrame:
    """Load encoded union for the specified labels from either combined CSV or per-label files."""
    if os.path.exists(ENCODED_PATH):
        df = pd.read_csv(ENCODED_PATH)
        label_ids = pre.encoders['label'].transform(labels)
        return df[df['Label'].isin(label_ids)].copy()
    # fallback: read per-label files
    frames = []
    for name in labels:
        safe = cic2018.get_label_name(name)
        path = os.path.join(ENCODED_DIR, f"cic2018_{safe}_encoded.csv")
        if not os.path.exists(path):
            logger.warning(f"[!] Missing encoded file for label {name}: {path}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No encoded data found for requested labels. Run encoding or splitting first.")
    return pd.concat(frames, ignore_index=True)


def _compress_single_label(pre: CIC2018Preprocessor, encoded_source_df: pd.DataFrame, label_name: str,
                           tau: int, extra_train: int = 2000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    lid = pre.encoders['label'].transform([label_name])[0]
    class_df = encoded_source_df[encoded_source_df['Label'] == lid].drop_duplicates()
    if class_df.empty:
        raise ValueError(f"No rows for label {label_name} in encoded source")
    # Compress directly to tau_effective = tau + extra_train using KMeans only
    tau_effective = int(tau) + int(extra_train)
    compressor = KMeansCompressor(tau=tau_effective)
    X = class_df.drop(columns=['Label'])
    y = class_df['Label']
    Xc, yc = compressor.compress_majority_class(X, y)
    comp_df = pd.concat([Xc, yc], axis=1).drop_duplicates()
    logger.info(f"[+] Compressed {label_name}: {len(class_df)} -> {len(comp_df)} (target {tau_effective})")

    # Custom split: train = floor(0.7 * tau) + extra_train, test = ceil(0.3 * tau)
    base_train = int(np.floor(0.7 * tau))
    base_test = int(tau - base_train)
    train_count = base_train + int(extra_train)
    test_count = base_test
    if train_count + test_count > len(comp_df):
        logger.warning(f"[!] Requested split {train_count}/{test_count} exceeds available {len(comp_df)}; adjusting train size")
        train_count = max(0, len(comp_df) - test_count)

    rng = np.random.RandomState(random_state)
    idx = np.arange(len(comp_df))
    rng.shuffle(idx)
    train_idx = idx[:train_count]
    test_idx = idx[train_count:train_count + test_count]
    train_df = comp_df.iloc[train_idx]
    test_df = comp_df.iloc[test_idx]
    logger.info(f"[+] {label_name} split -> train: {len(train_df)} (base {base_train} + extra {extra_train}), test: {len(test_df)} (base {base_test})")
    return train_df, test_df


def main():
    # Majority labels (fixed set as per existing logic)
    majority_labels = ['Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC',
                       'DoS attacks-Hulk', 'Bot', 'Infilteration',
                       'SSH-Bruteforce', 'DoS attacks-GoldenEye']
    
    parser = argparse.ArgumentParser(description="CIC2018 majority compression (KMeans-only, enlarged train for future ENN)")
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'], help='Compress all majority labels or a single label')
    parser.add_argument('--label', type=str, default=None, choices=majority_labels, help='Label name to compress when mode=label')
    parser.add_argument('--tau', type=int, default=20000, help='Target per-class size after KMeans compression (no ENN)')
    parser.add_argument('--extra-train', type=int, default=2000, help='Fixed extra samples to add to train split (train only)')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    tau = args.tau
    extra_train = args.extra_train

    

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.info("[+] Encoders not found. Setup encoders first (run cli.cic2018.encode_data)")

    if args.mode == 'label':
        if not args.label:
            raise SystemExit("--label is required when --mode label")
        if args.label not in majority_labels:
            logger.warning(f"[!] Provided label is not in predefined majority set: {args.label}")
        source_df = _load_encoded_union(pre, [args.label])
        tr_enc, te_enc = _compress_single_label(pre, source_df, args.label, tau=tau, extra_train=extra_train)

        # Save per-label encoded and raw
        safe = cic2018.get_label_name(args.label)
        os.makedirs(ENCODED_DIR, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        enc_train_path = os.path.join(ENCODED_DIR, f'cic2018_{safe}_majority_train_compressed_encoded.csv')
        tr_enc.to_csv(enc_train_path, index=False)
        logger.info(f"[+] Saved per-label encoded train: {enc_train_path}")

        # Only save raw for test; train raw will be generated after ENN refine
        raw_test = pre.inverse_transform(te_enc)
        raw_test_path = os.path.join(RAW_DIR, f'cic2018_{safe}_majority_test_compressed_raw.csv')
        raw_test.to_csv(raw_test_path, index=False)
        logger.info(f"[+] Saved per-label raw test: {raw_test_path}")

    else:
        logger.info(f"[+] Majority labels (> {tau} target): {majority_labels}")
        source_df = _load_encoded_union(pre, majority_labels)

        per_label_tr = []
        per_label_te = []
        os.makedirs(ENCODED_DIR, exist_ok=True)
        os.makedirs(RAW_DIR, exist_ok=True)

        for name in majority_labels:
            try:
                tr_enc, te_enc = _compress_single_label(pre, source_df, name, tau=tau, extra_train=extra_train)
            except Exception as e:
                logger.warning(f"[!] Skip {name}: {e}")
                continue

            safe = cic2018.get_label_name(name)
            # Save per-label encoded
            enc_train_path = os.path.join(ENCODED_DIR, f'cic2018_{safe}_majority_train_compressed_encoded.csv')
            tr_enc.to_csv(enc_train_path, index=False)
            # Save only per-label raw test; raw train will be generated post-ENN
            raw_test = pre.inverse_transform(te_enc)
            raw_test_path = os.path.join(RAW_DIR, f'cic2018_{safe}_majority_test_compressed_raw.csv')
            raw_test.to_csv(raw_test_path, index=False)

            per_label_tr.append(tr_enc)
            per_label_te.append(te_enc)
            logger.info(f"[+] Saved per-label outputs for {name}")

        # No combined outputs; per-label files only

    logger.info("[+] CIC majority compression completed")


if __name__ == "__main__":
    main()


