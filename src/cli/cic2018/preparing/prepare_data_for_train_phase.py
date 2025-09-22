import os
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from configs import cic2018

logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


OUT_TRAIN = os.path.join(DATA_FOLDER, "cic2018_final_train_balanced_cat_map.csv")


def main():
    parser = argparse.ArgumentParser(description="Prepare CIC2018 final train/test (ordinal-mapped) from merged raw_processed inputs")
    parser.add_argument("--train-in", type=str, default=os.path.join(DATA_FOLDER, "cic2018_merged_train_raw_processed.csv"),
                        help="Merged train raw_processed CSV (default: DATA_FOLDER/cic2018_merged_train_raw_processed.csv)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    final_train_path = args.train_in
    logger.info(f"[+] Loading merged train RAW from {final_train_path} …")

    if not (os.path.exists(final_train_path)):
        raise SystemExit(
            "Merged datasets not found. Build them via: python -m cli.cic2018.merge_major_minor first."
        )

    logger.info("[+] Loading final train …")
    train_df = pd.read_csv(final_train_path)
    logger.info(f"[+] Train: {train_df.shape}")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.error("[-] Encoders not found")
        raise SystemExit("Encoders not found. Run setup_encoders.py first.")
        
    else:
        # Ensure only expected features are present
        train_df = pre.select_features_and_label(train_df)

    # Ordinal-encode categorical features (Protocol) for classical models
    train_df_mapped = pre.preprocess_encode_ordinal_features(train_df.copy())

    # Binary encode binary features
    train_df_mapped = pre.preprocess_encode_binary_features(train_df_mapped)

    # Encode labels
    train_df_mapped = pre.preprocess_encode_label(train_df_mapped)

    # Optional: do not encode labels here (keep original strings) for readability
    # train_df_mapped = pre.preprocess_encode_label(train_df_mapped)

    # Drop duplicates
    train_df_mapped = pre.fix_duplicates(train_df_mapped)

    # Brief categorical stats
    cat_feats = pre.encoded_categorical_features_ordinal
    merged = pd.concat([train_df_mapped], ignore_index=True)
    logger.info("categorical features details (merged train):")
    for col in cat_feats:
        if col in merged.columns:
            logger.info(f"{col}: {merged[col].nunique()} unique; min={merged[col].min()}, max={merged[col].max()}")

    # Export
    train_df_mapped.to_csv(OUT_TRAIN, index=False)
    logger.info(f"[+] Saved: {OUT_TRAIN}")


if __name__ == "__main__":
    main()


