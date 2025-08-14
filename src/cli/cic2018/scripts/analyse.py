import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger


# Paths should match extract.py outputs
DEST_FOLDER = "/dis/DS/minhtq/CIC-2018/"
OUTPUT_FILENAME = "CIC2018_raw_processed.csv"


logger = get_logger(__name__)


def read_columns(csv_path: str) -> List[str]:
    header_df = pd.read_csv(csv_path, nrows=0, low_memory=False)
    return header_df.columns.tolist()


def sample_dtypes(csv_path: str, columns: List[str], sample_rows: int = 100_000) -> Dict[str, str]:
    # Use a moderately sized sample to infer dtypes
    try:
        sample_df = pd.read_csv(
            csv_path,
            usecols=columns,
            nrows=sample_rows,
            low_memory=False,
        )
    except Exception:
        # Fallback to a smaller sample
        sample_df = pd.read_csv(
            csv_path,
            usecols=columns,
            nrows=10_000,
            low_memory=False,
        )
    return {col: str(dtype) for col, dtype in sample_df.dtypes.items()}


def compute_label_distribution_and_count(
    csv_path: str, chunksize: int = 500_000
) -> Tuple[pd.Series, int]:
    total_rows = 0
    label_counts = pd.Series(dtype="int64")

    for chunk in pd.read_csv(
        csv_path,
        usecols=["Label"],
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk["Label"] = chunk["Label"].astype(str).str.strip()
        # Defensive: drop accidental header repeats
        chunk = chunk[chunk["Label"] != "Label"]
        total_rows += len(chunk)
        label_counts = label_counts.add(chunk["Label"].value_counts(), fill_value=0)

    label_counts = label_counts.fillna(0).astype(int).sort_values(ascending=False)
    return label_counts, total_rows


def compute_unique_counts(
    csv_path: str,
    columns: List[str],
    chunksize: int = 500_000,
    cap_per_column: int = 1_000_000,
) -> Dict[str, Tuple[int, bool]]:
    """
    Returns mapping: column -> (unique_count_or_lower_bound, capped_flag)
    If capped_flag is True, value is a lower bound (>=).
    Process columns sequentially to keep memory bounded.
    """
    unique_counts: Dict[str, Tuple[int, bool]] = {}

    for col in columns:
        logger.info(f"[+] Counting unique values for column: {col}")
        seen = set()
        capped = False

        for chunk in pd.read_csv(
            csv_path,
            usecols=[col],
            low_memory=False,
            chunksize=chunksize,
        ):
            # Replace inf with NaN then drop NaN (shouldn't exist in clean file, but defensive)
            if np.issubdtype(chunk[col].dtype, np.number):
                chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan)
            chunk = chunk.dropna(subset=[col])

            # Update set
            values = chunk[col].tolist()
            seen.update(values)

            if len(seen) > cap_per_column:
                capped = True
                # Trim by keeping only a subset to avoid unbounded memory growth
                # Keep first cap_per_column items deterministically
                if len(seen) > cap_per_column:
                    # Convert to list to slice then back to set
                    trimmed = list(seen)[:cap_per_column]
                    seen = set(trimmed)

        unique_counts[col] = (len(seen), capped)

    return unique_counts


if __name__ == "__main__":
    setup_logging("INFO")

    csv_path = os.path.join(DEST_FOLDER, OUTPUT_FILENAME)
    if not os.path.exists(csv_path):
        raise SystemExit(f"Cleaned CSV not found: {csv_path}")

    logger.info(f"[+] Analyzing file: {csv_path}")

    # 1) Columns
    columns = read_columns(csv_path)
    logger.info(f"[+] Columns ({len(columns)}): {columns}")

    # 2) Dtypes (sample-based)
    dtypes_map = sample_dtypes(csv_path, columns)
    logger.info("[+] Dtypes (sample-inferred):")
    for col in columns:
        logger.info(f"    {col}: {dtypes_map.get(col, 'unknown')}")

    # 3) Label distribution and total rows
    label_counts, total_rows = compute_label_distribution_and_count(csv_path)
    logger.info(f"[+] Total records: {total_rows}")
    logger.info("[+] Label distribution:")
    for label, cnt in label_counts.items():
        logger.info(f"    {label}: {cnt}")

    exit()
    # 4) Unique counts per column (sequential, memory-bounded with cap)
    unique_counts = compute_unique_counts(csv_path, columns)
    logger.info("[+] Unique values per column:")
    for col in columns:
        count, capped = unique_counts[col]
        suffix = " (>=, capped)" if capped else ""
        logger.info(f"    {col}: {count}{suffix}")


