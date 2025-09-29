import os
import argparse
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


def _list_clean_merged_files(input_dir: str) -> list[str]:
    files: list[str] = []
    if not os.path.isdir(input_dir):
        return files
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("cic2018_") and fname.endswith("_clean_merged.csv"):
            files.append(os.path.join(input_dir, fname))
    return files


def _analyze_file(fp: str, cont_cols: list[str], chunksize: int, thr_50: int, thr_70: int) -> tuple[int, int, int]:
    total_rows = 0
    cnt_ge_50 = 0
    cnt_ge_70 = 0
    usecols = [c for c in cont_cols if c]
    if not usecols:
        logger.warning(f"[!] No continuous columns present in file schema: {fp}")
        return 0, 0, 0

    for chunk in pd.read_csv(fp, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
        # Ensure numeric
        for c in usecols:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        zero_counts = (chunk[usecols] == 0).sum(axis=1)
        total_rows += len(chunk)
        cnt_ge_50 += int((zero_counts >= thr_50).sum())
        cnt_ge_70 += int((zero_counts >= thr_70).sum())

    return total_rows, cnt_ge_50, cnt_ge_70


def main():
    parser = argparse.ArgumentParser(description="Analyze zero-heavy rows in clean_merged per-label files (pre-split)")
    parser.add_argument('--input-dir', type=str, default=cic2018.CLEAN_MERGED_DATA_FOLDER,
                        help='Directory containing cic2018_<label>_clean_merged.csv files')
    parser.add_argument('--chunksize', type=int, default=500_000,
                        help='Read CSV in chunks of this many rows')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    input_dir = args.input_dir
    files = _list_clean_merged_files(input_dir)
    if not files:
        raise SystemExit(f"No clean_merged files found in {input_dir}")

    pre = CIC2018Preprocessor()
    cont_cols = pre.cont_features
    num_cont = len(cont_cols)
    thr_50 = 28  # ~50% of 57
    thr_70 = 40  # ~70% of 57
    logger.info(f"[+] Continuous features considered: {num_cont} | thresholds: >={thr_50} (≈50%), >={thr_70} (≈70%)")

    summary_rows: list[dict] = []
    for fp in files:
        label_safe = os.path.basename(fp)[len('cic2018_'):-len('_clean_merged.csv')]
        logger.info(f"[+] Analyzing: {os.path.basename(fp)}")
        total, ge50, ge70 = _analyze_file(fp, cont_cols, args.chunksize, thr_50, thr_70)
        logger.info(f"    total_rows={total} | rows_zero>={thr_50}={ge50} | rows_zero>={thr_70}={ge70}")
        summary_rows.append({
            'label': label_safe,
            'total_rows': int(total),
            f'rows_zero_gte_{thr_50}': int(ge50),
            f'rows_zero_gte_{thr_70}': int(ge70),
        })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    out_dir = os.path.join(cic2018.REPORT_FOLDER, 'anomaly')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'zero_heavy_rows_summary.csv')
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"[+] Summary saved -> {out_csv}")


if __name__ == '__main__':
    main()


