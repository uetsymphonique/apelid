import os
import sys
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import cic2018

logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


OUT_TEST  = os.path.join(DATA_FOLDER, "cic2018_final_test_cat_map.csv")


def main():
    parser = argparse.ArgumentParser(description="Prepare CIC2018 final test (ordinal-mapped) from merged raw_processed inputs")
    parser.add_argument("--test-in", type=str, default=os.path.join(DATA_FOLDER, "cic2018_merged_test_raw_processed.csv"),
                        help="Merged test raw_processed CSV (default: DATA_FOLDER/cic2018_merged_test_raw_processed.csv)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    final_test_path = args.test_in
    logger.info(f"[+] Loading merged test RAW from {final_test_path}")
    if not (os.path.exists(final_test_path)):
        raise SystemExit(
            "Merged datasets not found. Build them via: python -m cli.cic2018.merge_major_minor first."
        )

    logger.info("[+] Loading final test")
    test_df = pd.read_csv(final_test_path)
    logger.info(f"[+] Test: {test_df.shape}")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.error("[-] Encoders not found")
        raise SystemExit("Encoders not found. Run setup_encoders.py first.")
    else:
        # Ensure only expected features are present
        test_df = pre.select_features_and_label(test_df)

    # Ordinal-encode categorical features (Protocol) for classical models
    test_df_mapped  = pre.preprocess_encode_ordinal_features(test_df.copy())

    # Binary encode binary features
    test_df_mapped  = pre.preprocess_encode_binary_features(test_df_mapped)

    # Encode labels
    test_df_mapped  = pre.preprocess_encode_label(test_df_mapped)

    # Optional: do not encode labels here (keep original strings) for readability
    # test_df_mapped  = pre.preprocess_encode_label(test_df_mapped)

    # Drop duplicates
    test_df_mapped  = pre.fix_duplicates(test_df_mapped)

    # Brief categorical stats
    cat_feats = pre.encoded_categorical_features_ordinal
    merged = pd.concat([test_df_mapped], ignore_index=True)
    logger.info("categorical features details (merged test):")
    for col in cat_feats:
        if col in merged.columns:
            logger.info(f"{col}: {merged[col].nunique()} unique; min={merged[col].min()}, max={merged[col].max()}")

    # Label distribution (original label names)
    try:
        le = pre.encoders.get('label')
        if le is not None and 'Label' in test_df_mapped.columns:
            labels_str = le.inverse_transform(test_df_mapped['Label'].values)
            vc = pd.Series(labels_str).value_counts()
            logger.info("label distribution (original names):")
            for name, cnt in vc.items():
                logger.info(f"  - {name}: {int(cnt)}")
    except Exception as e:
        logger.warning(f"[!] Could not log label distribution (original names): {e}")
    
    # Export
    test_df_mapped.to_csv(OUT_TEST, index=False)
    logger.info(f"[+] Saved: {OUT_TEST}")


if __name__ == "__main__":
    main()


