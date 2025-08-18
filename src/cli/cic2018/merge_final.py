import os
import argparse
import pandas as pd
from sklearn.utils import shuffle

from utils.logging import get_logger, setup_logging
from configs import cic2018

logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER
def merge_test_raw() -> str:
    logger.info("[+] Merging per-label test RAW (majority + minority) …")
    test_files = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith('_majority_test_compressed_raw.csv') or fname.endswith('_minority_test_raw.csv'):
            test_files.append(os.path.join(RAW_DIR, fname))
    if not test_files:
        raise SystemExit(f"No per-label test RAW files found in {RAW_DIR}.")
    frames = []
    for fp in test_files:
        try:
            df = pd.read_csv(fp)
            frames.append(df)
        except Exception as e:
            logger.warning(f"[!] Skip test file {fp}: {e}")
    final_test = pd.concat(frames, ignore_index=True)
    final_test = shuffle(final_test, random_state=42).reset_index(drop=True)
    out_path = os.path.join(DATA_FOLDER, 'cic2018_final_test_raw.csv')
    final_test.to_csv(out_path, index=False)
    logger.info(f"[+] Saved merged test RAW: {out_path} ({len(final_test)})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Merge only test RAW; train RAW is produced by ENN refiner")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    test_path = merge_test_raw()
    logger.info(f"[+] Done. Merged test RAW -> {test_path}")


if __name__ == "__main__":
    main()


