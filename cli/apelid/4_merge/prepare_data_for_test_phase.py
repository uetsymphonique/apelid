import os
import sys
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def main():
    parser = argparse.ArgumentParser(description="Prepare final TEST (ordinal-mapped) from merged raw_processed inputs (multi-resource)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--test-in", type=str, required=True,
                        help="Merged test raw_processed CSV (default: DATA_FOLDER/<resource>_merged_test_raw_processed.csv)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    final_test_path = args.test_in
    out_test  = os.path.join(res.DATA_FOLDER, f"{res.resources_name}_final_test_cat_map.csv")
    logger.info(f"[+] Loading merged test RAW from {final_test_path}")
    if not (os.path.exists(final_test_path)):
        raise SystemExit(
            f"Merged datasets not found. Build them via: python -m cli.cic2018.preparing.merge_major_minor_multi -r {res.resources_name} first."
        )

    logger.info("[+] Loading final test")
    test_df = pd.read_csv(final_test_path)
    logger.info(f"[+] Test: {test_df.shape}")

    if not pre.load_encoders():
        logger.error("[-] Encoders not found")
        raise SystemExit("Encoders not found. Run setup_encoders.py first.")
    else:
        test_df = pre.select_features_and_label(test_df)

    # Ordinal-encode categorical features (if available)
    if hasattr(pre, 'preprocess_encode_ordinal_features'):
        test_df_mapped  = pre.preprocess_encode_ordinal_features(test_df.copy())
    else:
        test_df_mapped = test_df.copy()

    # Binary encode binary features
    if hasattr(pre, 'preprocess_encode_binary_features'):
        test_df_mapped  = pre.preprocess_encode_binary_features(test_df_mapped)

    # Encode labels
    if hasattr(pre, 'preprocess_encode_label'):
        test_df_mapped  = pre.preprocess_encode_label(test_df_mapped)

    # Drop duplicates if available
    if hasattr(pre, 'fix_duplicates'):
        test_df_mapped  = pre.fix_duplicates(test_df_mapped)

    # Brief categorical stats
    cat_feats = getattr(pre, 'encoded_categorical_features_ordinal', [])
    if cat_feats:
        merged = pd.concat([test_df_mapped], ignore_index=True)
        logger.info("categorical features details (merged test):")
        for col in cat_feats:
            if col in merged.columns:
                logger.info(f"{col}: {merged[col].nunique()} unique; min={merged[col].min()}, max={merged[col].max()}")

    # Label distribution (original label names)
    try:
        le = getattr(pre, 'encoders', {}).get('label') if hasattr(pre, 'encoders') else None
        if le is not None and 'Label' in test_df_mapped.columns:
            labels_str = le.inverse_transform(test_df_mapped['Label'].values)
            vc = pd.Series(labels_str).value_counts()
            logger.info("label distribution (original names):")
            for name, cnt in vc.items():
                logger.info(f"  - {name}: {int(cnt)}")
    except Exception as e:
        logger.warning(f"[!] Could not log label distribution (original names): {e}")
    
    # Export
    test_df_mapped.to_csv(out_test, index=False)
    logger.info(f"[+] Saved: {out_test}")


if __name__ == "__main__":
    main()


