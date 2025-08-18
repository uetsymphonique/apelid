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
OUT_TEST  = os.path.join(DATA_FOLDER, "cic2018_final_test_cat_map.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minority-strategy", type=str, default="wgan", choices=["wgan", "cfm", "fdm"])
    parser.add_argument("--refine-mode", type=str, default="global", choices=["global", "pairwise"],
                        help="Select refined train source: global ENN or pairwise+global ENN")
    parser.add_argument("--pair-a", type=str, default="Benign", help="Class A name for pairwise refine (when --refine-mode=pairwise)")
    parser.add_argument("--pair-b", type=str, default="Infilteration", help="Class B name for pairwise refine (when --refine-mode=pairwise)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Updated inputs: refined train RAW (by ENN) and merged test RAW
    if args.refine_mode == "global":
        final_train_path = os.path.join(RAW_DIR, f"cic2018_train_refined_{args.minority_strategy}_raw.csv")
    else:
        a_safe = cic2018.get_label_name(args.pair_a)
        b_safe = cic2018.get_label_name(args.pair_b)
        final_train_path = os.path.join(RAW_DIR, f"cic2018_train_refined_pair_{a_safe}_{b_safe}_{args.minority_strategy}_raw.csv")
    final_test_path  = os.path.join(DATA_FOLDER, "cic2018_final_test_raw.csv")
    logger.info(f"[+] Loading refined train RAW and merged test RAW from {final_train_path} and {final_test_path} …")

    if not (os.path.exists(final_train_path) and os.path.exists(final_test_path)):
        raise SystemExit(
            "Final datasets not found. Run 'python -m cli.cic2018.refine_with_enn' and 'python -m cli.cic2018.merge_final' first."
        )

    logger.info("[+] Loading final train/test …")
    train_df = pd.read_csv(final_train_path)
    test_df = pd.read_csv(final_test_path)
    logger.info(f"[+] Train: {train_df.shape}, Test: {test_df.shape}")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.info("[+] Encoders not found, fitting on train set …")
        # Ensure selection matches training features
        train_df = pre.select_features_and_label(train_df)
        test_df = pre.select_features_and_label(test_df)
        pre.setup_encoders(train_df)
        pre.save_encoders()
    else:
        # Ensure only expected features are present
        train_df = pre.select_features_and_label(train_df)
        test_df = pre.select_features_and_label(test_df)

    # Ordinal-encode categorical features (Protocol) for classical models
    train_df_mapped = pre.preprocess_encode_ordinal_features(train_df.copy())
    test_df_mapped  = pre.preprocess_encode_ordinal_features(test_df.copy())

    # Optional: do not encode labels here (keep original strings) for readability
    # train_df_mapped = pre.preprocess_encode_label(train_df_mapped)
    # test_df_mapped  = pre.preprocess_encode_label(test_df_mapped)

    # Drop duplicates
    train_df_mapped = pre.fix_duplicates(train_df_mapped)
    test_df_mapped  = pre.fix_duplicates(test_df_mapped)

    # Brief categorical stats
    cat_feats = pre.encoded_categorical_features_ordinal
    merged = pd.concat([train_df_mapped, test_df_mapped], ignore_index=True)
    logger.info("categorical features details (merged train and test):")
    for col in cat_feats:
        if col in merged.columns:
            logger.info(f"{col}: {merged[col].nunique()} unique; min={merged[col].min()}, max={merged[col].max()}")

    # Export
    train_df_mapped.to_csv(OUT_TRAIN, index=False)
    test_df_mapped.to_csv(OUT_TEST, index=False)
    logger.info(f"[+] Saved: {OUT_TRAIN}")
    logger.info(f"[+] Saved: {OUT_TEST}")


if __name__ == "__main__":
    main()


