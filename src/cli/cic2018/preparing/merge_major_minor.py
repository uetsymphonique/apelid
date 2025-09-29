import os
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


DATA_FOLDER = cic2018.DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER
CLEAN_DIR = cic2018.CLEAN_MERGED_DATA_FOLDER


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    # Union of columns, then reindex each
    cols = set()
    for d in dfs:
        cols.update(d.columns.tolist())
    ordered = sorted(cols)
    normed = [d.reindex(columns=ordered) for d in dfs]
    return pd.concat(normed, ignore_index=True)


def _path_major_train_clean_merged_compressed(label: str) -> str:
    safe = cic2018.get_label_name(label)
    return os.path.join(CLEAN_DIR, 'train', f"cic2018_{safe}_train_clean_merged_compressed.csv")





def _path_major_train_clean_merged(label: str) -> str:
    safe = cic2018.get_label_name(label)
    # Produced by split_clean_merged.py
    return os.path.join(CLEAN_DIR, 'train', f"cic2018_{safe}_train_clean_merged.csv")


def _path_major_test_clean_merged(label: str) -> str:
    safe = cic2018.get_label_name(label)
    # Produced by split_clean_merged.py
    return os.path.join(CLEAN_DIR, 'test', f"cic2018_{safe}_test_clean_merged.csv")


def _path_major_test_clean_merged_compressed(label: str) -> str:
    safe = cic2018.get_label_name(label)
    return os.path.join(CLEAN_DIR, 'test', f"cic2018_{safe}_test_clean_merged_compressed.csv")


def _path_minor_train_augmented(label: str, strategy: str) -> str:
    safe = cic2018.get_label_name(label)
    return os.path.join(RAW_DIR, 'train', f"cic2018_{safe}_minority_{strategy}_train_augmented_raw_processed.csv")


def _path_minor_test_clean(label: str) -> str:
    safe = cic2018.get_label_name(label)
    return os.path.join(CLEAN_DIR, 'test', f"cic2018_{safe}_test_clean_merged.csv")


def main():
    parser = argparse.ArgumentParser(description="Merge CIC2018 datasets into single train/test CSVs | Major: clean_merged_compressed (train/test); Minor: train augmented raw_processed, test clean_merged")
    parser.add_argument('--strategy', type=str, default='wgan', choices=['wgan', 'cfm', 'fdm'],
                        help='Augment strategy name used for minor train filenames (default: wgan)')
    parser.add_argument('--train-out', type=str, default=os.path.join(DATA_FOLDER, 'cic2018_merged_train_raw_processed.csv'))
    parser.add_argument('--test-out', type=str, default=os.path.join(DATA_FOLDER, 'cic2018_merged_test_raw_processed.csv'))
    # Major sources are now fixed: prefer clean_merged compressed, fallback to original clean_merged
    parser.add_argument('--subset', type=str, default='both', choices=['both', 'train', 'test'],
                        help='Which subset(s) to merge: both (default), train only, or test only')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.subset in ('both', 'train'):
        logger.info("[+] Merging TRAIN: major clean_merged_compressed + minor augmented raw_processed …")
        train_parts: list[pd.DataFrame] = []
        # Major train: ONLY clean_merged compressed
        for lbl in cic2018.MAJORITY_LABELS:
            p_comp = _path_major_train_clean_merged_compressed(lbl)
            if os.path.exists(p_comp):
                try:
                    logger.debug(f"[+] Loading major train compressed: {p_comp}")
                    df = pd.read_csv(p_comp, low_memory=False)
                    df['__source__'] = f"major:{lbl}"
                    train_parts.append(df)
                    logger.info(f"[+] Major train OK: {lbl} ({len(df)}) | source=compressed")
                except Exception as e:
                    logger.warning(f"[!] Skip major train {lbl}: {e}")
            else:
                logger.warning(f"[-] Missing major train compressed: {p_comp}")
        # Minor train (augmented)
        for lbl in cic2018.MINORITY_LABELS:
            p = _path_minor_train_augmented(lbl, args.strategy)
            if os.path.exists(p):
                try:
                    logger.debug(f"[+] Loading minor train augmented: {p}")
                    df = pd.read_csv(p, low_memory=False)
                    df['__source__'] = f"minor:{lbl}:{args.strategy}"
                    train_parts.append(df)
                    logger.info(f"[+] Minor train OK: {lbl} ({len(df)})")
                except Exception as e:
                    logger.warning(f"[!] Skip minor train {lbl}: {e}")
            else:
                logger.debug(f"[-] Missing minor train: {p}")

        train_merged = _safe_concat(train_parts)
        if len(train_merged) == 0:
            logger.error("[!] No train parts found; nothing to merge")
        else:
            # Shuffle for safety (reproducible)
            train_merged = train_merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
            logger.info("[+] Shuffled merged TRAIN (seed=42)")
            os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
            train_merged.to_csv(args.train_out, index=False)
            logger.info(f"[+] Saved merged TRAIN -> {args.train_out} ({len(train_merged)})")

    if args.subset in ('both', 'test'):
        logger.info("[+] Merging TEST: major clean_merged_compressed + minor test clean_merged …")
        test_parts: list[pd.DataFrame] = []
        # Major test: ONLY clean_merged compressed
        for lbl in cic2018.MAJORITY_LABELS:
            p_comp = _path_major_test_clean_merged_compressed(lbl)
            if os.path.exists(p_comp):
                try:
                    logger.debug(f"[+] Loading major test compressed: {p_comp}")
                    df = pd.read_csv(p_comp, low_memory=False)
                    df['__source__'] = f"major:{lbl}"
                    test_parts.append(df)
                    logger.info(f"[+] Major test OK: {lbl} ({len(df)}) | source=compressed")
                except Exception as e:
                    logger.warning(f"[!] Skip major test {lbl}: {e}")
            else:
                logger.warning(f"[-] Missing major test compressed: {p_comp}")
        # Minor test (clean_merged)
        for lbl in cic2018.MINORITY_LABELS:
            p = _path_minor_test_clean(lbl)
            if os.path.exists(p):
                try:
                    logger.debug(f"[+] Loading minor test clean_merged: {p}")
                    df = pd.read_csv(p, low_memory=False)
                    df['__source__'] = f"minor:{lbl}:clean_merged"
                    test_parts.append(df)
                    logger.info(f"[+] Minor test OK: {lbl} ({len(df)})")
                except Exception as e:
                    logger.warning(f"[!] Skip minor test {lbl}: {e}")
            else:
                logger.debug(f"[-] Missing minor test: {p}")

        test_merged = _safe_concat(test_parts)
        if len(test_merged) == 0:
            logger.error("[!] No test parts found; nothing to merge")
        else:
            # Shuffle for safety (reproducible)
            test_merged = test_merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
            logger.info("[+] Shuffled merged TEST (seed=42)")
            os.makedirs(os.path.dirname(args.test_out), exist_ok=True)
            test_merged.to_csv(args.test_out, index=False)
            logger.info(f"[+] Saved merged TEST -> {args.test_out} ({len(test_merged)})")


if __name__ == "__main__":
    main()


