import os
import argparse
import pandas as pd

import configs.cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from utils.logging import setup_logging, get_logger


logger = get_logger(__name__)


CLEAN_MERGED_DIR = configs.cic2018.CLEAN_MERGED_DATA_FOLDER


def main():
    parser = argparse.ArgumentParser(description="Setup CIC-IDS2018 encoders using TRAIN split only (to avoid leakage)")
    parser.add_argument("--input-dir", type=str, default=CLEAN_MERGED_DIR,
                        help="Root directory of clean merged data containing 'train' subfolder")
    parser.add_argument("--train-subdir", type=str, default="train",
                        help="Subdirectory under input-dir that holds per-label TRAIN files")
    parser.add_argument("--sentinel-impute", type=str, default="none", choices=["median", "zero", "none"],
                        help="Handle -1 sentinel in Init Fwd/Bwd Win Byts before fitting encoders (default zero)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()
    setup_logging(args.log_level)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    logger.info("===========================================")
    logger.info("ENCODER SETUP STARTED")
    logger.info("===========================================")
    train_dir = os.path.join(input_dir, args.train_subdir)
    if not os.path.isdir(train_dir):
        raise SystemExit(f"TRAIN directory not found: {train_dir}. Run 'split_clean_merged.py' first.")

    logger.info(f"Scanning TRAIN per-label files in: {train_dir}")

    label_files = [
        os.path.join(train_dir, f)
        for f in sorted(os.listdir(train_dir))
        if f.endswith('_train_clean_merged.csv') and f.startswith('cic2018_')
    ]
    if not label_files:
        raise SystemExit(f"No TRAIN files found in {train_dir}. Expected cic2018_<label>_train_clean_merged.csv")

    logger.info(f"Found {len(label_files)} TRAIN label files")

    # Initialize preprocessor early for cleaning pipeline
    preprocessor = CIC2018Preprocessor()

    # Load and concatenate all per-label data to fit encoders (ensure consistent schema)
    union_frames = []
    total_rows = 0
    for lf in label_files:
        df_part = pd.read_csv(lf, low_memory=False)
        # Align to expected features and coerce dtypes
        df_part = preprocessor.select_features_and_label(df_part)
        df_part = preprocessor.coerce_feature_dtypes(df_part)
        # Sentinel handling (-1) before fitting encoders to avoid distribution skew
        if args.sentinel_impute != "none":
            fill_val = 0.0 if args.sentinel_impute == "zero" else 0.0
            df_part = preprocessor.add_sentinel_indicators_and_impute_init_win_bytes(
                df_part, strategy=("median" if args.sentinel_impute == "median" else "fixed"), fill_value=fill_val
            )
        # Remove NaN/Inf and negatives (excluding sentinel columns inside preprocessor)
        # df_part = preprocessor.remove_missing_and_inf_values(df_part)
        # df_part = preprocessor.remove_negative_numeric_rows(df_part)
        union_frames.append(df_part)
        total_rows += len(df_part)
    union_df = pd.concat(union_frames, ignore_index=True)
    logger.info(f"TRAIN union data for encoder fit: {union_df.shape} (from {len(label_files)} files, {total_rows} rows)")

    # Fit and persist encoders
    preprocessor.setup_encoders(union_df)
    preprocessor.save_encoders()

    logger.info("Encoders have been fitted on TRAIN and saved.")
    logger.info("===========================================")
    logger.info("ENCODER SETUP COMPLETED")
    logger.info("===========================================")


if __name__ == "__main__":
    main()


