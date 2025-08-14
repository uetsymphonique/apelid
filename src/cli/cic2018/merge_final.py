import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from datasvc.data_service import DataService


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"

MAJ_TRAIN_RAW = os.path.join(DATA_FOLDER, "cic_majority_train_compressed_raw.csv")
MAJ_TEST_RAW  = os.path.join(DATA_FOLDER, "cic_majority_test_compressed_raw.csv")
MIN_TRAIN_RAW = os.path.join(DATA_FOLDER, "cic_minority_train_augmented_raw.csv")
MIN_TEST_RAW  = os.path.join(DATA_FOLDER, "cic_minority_test_raw.csv")

FINAL_TRAIN = os.path.join(DATA_FOLDER, "cic_final_train_balanced.csv")
FINAL_TEST  = os.path.join(DATA_FOLDER, "cic_final_test.csv")


def merge_final_datasets():
    logger.info("[+] Starting CIC final dataset merge…")

    # Load majority data
    try:
        maj_train = pd.read_csv(MAJ_TRAIN_RAW)
        maj_test  = pd.read_csv(MAJ_TEST_RAW)
        logger.info(f"[+] Loaded majority: train={len(maj_train)}, test={len(maj_test)}")
    except FileNotFoundError as e:
        logger.error(f"[-] Majority data not found: {e}. Run majority_compression first.")
        return

    # Load minority data
    try:
        min_train = pd.read_csv(MIN_TRAIN_RAW)
        min_test  = pd.read_csv(MIN_TEST_RAW)
        logger.info(f"[+] Loaded minority: train={len(min_train)}, test={len(min_test)}")
    except FileNotFoundError as e:
        logger.error(f"[-] Minority data not found: {e}. Run minority_wgan first.")
        return

    # Initialize preprocessor for info only
    pre = CIC2018Preprocessor()

    # Show distributions before merge
    logger.info("[+] Majority train distribution:")
    logger.info(maj_train['Label'].value_counts())
    logger.info("[+] Minority train distribution:")
    logger.info(min_train['Label'].value_counts())

    # Merge
    final_train = pd.concat([maj_train, min_train], ignore_index=True)
    final_train = shuffle(final_train, random_state=42).reset_index(drop=True)

    final_test = pd.concat([maj_test, min_test], ignore_index=True)
    final_test = shuffle(final_test, random_state=42).reset_index(drop=True)

    # Clean: duplicates, inf -> NaN, missing values
    def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
        svc = DataService(df)
        rows_before = len(svc.df)
        # Duplicates
        if svc.check_duplicates():
            logger.info(f"[+] {name}: duplicates detected – dropping …")
        svc.fix_duplicates()
        rows_after_dup = len(svc.df)
        if rows_after_dup != rows_before:
            logger.info(f"[+] {name}: dropped {rows_before - rows_after_dup} duplicate rows")
        # Infinity -> NaN
        svc.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Missing
        if svc.check_missing_values():
            missing_before = svc.df.isna().sum().sum()
            # DataService.fix_missing_values() doesn't assign; handle explicitly
            svc.df = svc.df.dropna()
            logger.info(f"[+] {name}: dropped rows with NaN/Inf (total NaNs before: {missing_before})")
        return svc.df

    final_train = clean_df(final_train, "Final Train")
    final_test  = clean_df(final_test,  "Final Test")

    # Report
    logger.info("[+] Final training dataset:")
    pre.info_dataset(final_train)
    logger.info("[+] Final test dataset:")
    pre.info_dataset(final_test)

    # Save
    final_train.to_csv(FINAL_TRAIN, index=False)
    final_test.to_csv(FINAL_TEST, index=False)
    logger.info("[+] Final datasets saved:")
    logger.info(f"    - {FINAL_TRAIN} ({len(final_train)} samples)")
    logger.info(f"    - {FINAL_TEST} ({len(final_test)} samples)")


if __name__ == "__main__":
    setup_logging("INFO")
    merge_final_datasets()


