import os
import pandas as pd
from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from resampling.undersampling.kmeans import KMeansCompressor
from resampling.undersampling.enn_refiner import ENNRefiner
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
ENCODED_PATH = os.path.join(DATA_FOLDER, "CIC2018_encoded.csv")
RAW_PATH = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")
UNIFIED_CLEAN_PATH = os.path.join(DATA_FOLDER, "CIC2018_unified_clean.csv")


def split_train_test(df, test_size=0.3, random_state=42):
    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


def main():
    setup_logging("INFO")

    # 1) Determine majority classes (> tau)
    tau = 20000
    tau_kmeans = 22000  # compress with slack, ENN will refine to exact tau
    majority_labels = ['Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 
                       'DoS attacks-Hulk', 'Bot', 'Infilteration', 
                       'SSH-Bruteforce', 'DoS attacks-GoldenEye']
    logger.info(f"[+] Majority labels (> {tau}): {majority_labels}")

    # 2) Load raw processed for inverse mapping and encoded for compression
    if not os.path.exists(ENCODED_PATH):
        raise SystemExit(f"Encoded dataset not found: {ENCODED_PATH}. Run cli-cic.preprocessing first.")
    encoded_df = pd.read_csv(ENCODED_PATH)
    logger.info(f"[+] Loaded encoded dataset: {encoded_df.shape}")

    # 3) Initialize preprocessor and load encoders
    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.info("[+] Encoders not found. Fitting on unified clean dataset…")
        if os.path.exists(UNIFIED_CLEAN_PATH):
            clean_df = pd.read_csv(UNIFIED_CLEAN_PATH)
            pre.setup_encoders(clean_df)  # No need for feature selection, already done
            pre.save_encoders()
        elif os.path.exists(RAW_PATH):
            raw_df = pd.read_csv(RAW_PATH)
            raw_df = pre.select_features_and_label(raw_df)
            pre.setup_encoders(raw_df)
            pre.save_encoders()
        else:
            raise FileNotFoundError("No suitable dataset found for encoder fitting")

    # 4) Encode label already numeric; ensure type
    encoded_df['Label'] = encoded_df['Label'].astype(int)

    # 5) Filter to majority classes (by original string names → convert to encoded ids)
    label_ids = pre.encoders['label'].transform(majority_labels)
    maj_df = encoded_df[encoded_df['Label'].isin(label_ids)].copy()
    # Note: Since we're using unified clean dataset, minimal duplicates expected
    # Only check for any encoding-induced duplicates
    before = len(maj_df)
    maj_df = maj_df.drop_duplicates()
    if len(maj_df) != before:
        logger.info(f"[+] Found {before - len(maj_df)} encoding-induced duplicates (removed)")
    else:
        logger.info(f"[+] No duplicates found in majority subset (as expected from unified clean data)")
    logger.info(f"[+] Majority subset: {maj_df.shape}")

    # 6) Compress each majority class to tau with KMeans, then ENN refine to exact tau
    compressor = KMeansCompressor(tau=tau_kmeans, use_enn=False)
    compressed_frames = []
    for lid in label_ids:
        class_df = maj_df[maj_df['Label'] == lid].drop_duplicates()
        X = class_df.drop(columns=['Label'])
        y = class_df['Label']
        Xc, yc = compressor.compress_majority_class(X, y)
        comp_df = pd.concat([Xc, yc], axis=1).drop_duplicates()
        compressed_frames.append(comp_df)
        logger.info(f"[+] Compressed label {lid}: {len(class_df)} -> {len(comp_df)} (target {tau_kmeans})")

    compressed_majority = pd.concat(compressed_frames, ignore_index=True)

    # ENN global refinement to ensure exactly tau per class
    refiner = ENNRefiner(tau_final=tau)
    compressed_majority = refiner.refine(compressed_majority, label_col='Label')

    # 7) Split 70/30
    train_df, test_df = split_train_test(compressed_majority, test_size=0.3, random_state=42)

    # 8) Save encoded train/test
    enc_train_path = os.path.join(DATA_FOLDER, 'cic_majority_train_compressed_encoded.csv')
    enc_test_path = os.path.join(DATA_FOLDER, 'cic_majority_test_compressed_encoded.csv')
    train_df.to_csv(enc_train_path, index=False)
    test_df.to_csv(enc_test_path, index=False)
    logger.info(f"[+] Saved encoded: {enc_train_path}, {enc_test_path}")

    # 9) Inverse to raw and save (using inverse transform for compatibility with downstream)
    raw_train = pre.inverse_transform(train_df)
    raw_test = pre.inverse_transform(test_df)
    raw_train_path = os.path.join(DATA_FOLDER, 'cic_majority_train_compressed_raw.csv')
    raw_test_path = os.path.join(DATA_FOLDER, 'cic_majority_test_compressed_raw.csv')
    raw_train.to_csv(raw_train_path, index=False)
    raw_test.to_csv(raw_test_path, index=False)
    logger.info(f"[+] Saved raw: {raw_train_path}, {raw_test_path}")
    logger.info(f"[+] Note: For original clean data, use {UNIFIED_CLEAN_PATH}")

    logger.info("[+] CIC majority compression completed")


if __name__ == "__main__":
    main()


