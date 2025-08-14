import os
import subprocess
import configs.cic2018
from collections import Counter
from typing import List, Tuple

import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger


data_folder = "/dis/DS/CIC2018"
dest_merged_folder = configs.cic2018.DATA_FOLDER


logger = get_logger(__name__)


def read_header_columns(csv_path: str) -> List[str]:
    header_df = pd.read_csv(
        csv_path, nrows=0, low_memory=False, skipinitialspace=True
    )
    return [c.strip() for c in header_df.columns.tolist()]


def detect_majority_schema(csv_files: List[str]) -> Tuple[List[str], dict]:
    schema_counter: Counter = Counter()
    file_to_schema = {}

    for fp in csv_files:
        cols = read_header_columns(fp)
        file_to_schema[fp] = cols
        schema_counter[tuple(cols)] += 1

    if not schema_counter:
        raise RuntimeError("No CSV headers detected in the provided folder")

    majority_schema_tuple, _ = schema_counter.most_common(1)[0]
    majority_schema = list(majority_schema_tuple)

    # Log outliers (files whose schema differs from majority)
    outliers = [
        fp for fp, cols in file_to_schema.items() if cols != majority_schema
    ]
    if outliers:
        logger.info(f"[+] Detected majority schema across {len(csv_files) - len(outliers)} files")
        logger.info("[+] Files with differing schema (will be aligned by dropping extras):")
        for fp in outliers:
            extra = [c for c in file_to_schema[fp] if c not in majority_schema]
            missing = [c for c in majority_schema if c not in file_to_schema[fp]]
            logger.info(f"    - {os.path.basename(fp)}: +{len(extra)} extras, -{len(missing)} missing")
    else:
        logger.info(f"[+] All {len(csv_files)} files share the same schema")

    return majority_schema, file_to_schema


def compute_label_distribution_and_column_count(
    csv_files: List[str], majority_schema: List[str], chunksize: int = 500_000
):
    label_counts = pd.Series(dtype="int64")

    for fp in csv_files:
        logger.info(f"[+] Processing: {os.path.basename(fp)}")
        # Read using the majority schema only; this drops any extra columns safely
        for chunk in pd.read_csv(
            fp,
            usecols=majority_schema,
            low_memory=False,
            skipinitialspace=True,
            chunksize=chunksize,
        ):
            if "Label" not in chunk.columns:
                raise RuntimeError(
                    f"Missing 'Label' column in file {fp} after schema alignment"
                )
            # Normalize label whitespace
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            # Drop repeated header rows that sometimes appear inside CIC-2018 CSVs
            before = len(chunk)
            chunk = chunk[chunk["Label"] != "Label"]
            dropped = before - len(chunk)
            if dropped:
                logger.debug(f"[+] Dropped {dropped} repeated header rows in {os.path.basename(fp)}")
            label_counts = label_counts.add(chunk["Label"].value_counts(), fill_value=0)

    # Ensure integer dtype for counts
    label_counts = label_counts.fillna(0).astype(int).sort_values(ascending=False)

    num_columns = len(majority_schema)
    return num_columns, label_counts


def merge_and_clean_to_csv(
    csv_files: List[str],
    majority_schema: List[str],
    dest_folder: str,
    output_filename: str = "CIC2018_merged_clean.csv",
    chunksize: int = 500_000,
):
    os.makedirs(dest_folder, exist_ok=True)
    output_path = os.path.join(dest_folder, output_filename)

    # Remove output if exists for a clean run
    if os.path.exists(output_path):
        os.remove(output_path)

    total_read = 0
    total_written = 0
    dropped_header_rows = 0
    dropped_missing_inf = 0
    dropped_duplicates = 0

    first_chunk = True

    for fp in csv_files:
        logger.info(f"[+] Writing cleaned data from: {os.path.basename(fp)}")

        for chunk in pd.read_csv(
            fp,
            usecols=majority_schema,
            low_memory=False,
            skipinitialspace=True,
            chunksize=chunksize,
        ):
            total_read += len(chunk)

            # Normalize and drop repeated header rows
            if "Label" not in chunk.columns:
                raise RuntimeError(
                    f"Missing 'Label' column in file {fp} after schema alignment"
                )
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            before = len(chunk)
            chunk = chunk[chunk["Label"] != "Label"]
            dropped_header_rows += before - len(chunk)

            # Replace +/- inf with NaN then drop any missing values
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            before = len(chunk)
            chunk = chunk.dropna(how="any")
            dropped_missing_inf += before - len(chunk)

            # Drop duplicates within chunk to reduce redundancy
            before = len(chunk)
            chunk = chunk.drop_duplicates()
            dropped_duplicates += before - len(chunk)

            if len(chunk) == 0:
                continue

            # Write to CSV in append mode after first write
            chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            total_written += len(chunk)

    logger.info(f"[+] Clean write completed: {output_path}")
    return {
        "output_path": output_path,
        "total_read": int(total_read),
        "total_written": int(total_written),
        "dropped_header_rows": int(dropped_header_rows),
        "dropped_missing_inf": int(dropped_missing_inf),
        "dropped_duplicates": int(dropped_duplicates),
    }


def _run_cmd_capture(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}\n{result.stderr}")
    return result.stdout.strip()


def verify_and_global_dedupe(input_csv: str, dest_folder: str) -> dict:
    """
    Verify duplicates across entire file, and if any, produce a globally de-duplicated file
    using external sort -u (disk-based). Returns stats including whether dedupe occurred and
    the path to the deduped file (or original if none needed).
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    # Count total lines (including header)
    total_lines_str = _run_cmd_capture(f"wc -l < '{input_csv}'")
    total_lines = int(total_lines_str)

    # Count unique data lines (excluding header) via sort -u
    unique_data_lines_str = _run_cmd_capture(
        f"tail -n +2 '{input_csv}' | sort -T /tmp -u | wc -l"
    )
    unique_data_lines = int(unique_data_lines_str)

    data_lines = max(total_lines - 1, 0)
    duplicates = max(data_lines - unique_data_lines, 0)

    if duplicates == 0:
        # No dedupe needed
        return {
            "dedup_performed": False,
            "input_path": input_csv,
            "output_path": input_csv,
            "total_lines": total_lines,
            "unique_data_lines": unique_data_lines,
            "duplicates": 0,
        }

    # Perform global dedupe: write header + unique sorted data lines
    dedup_path = os.path.join(dest_folder, "CIC2018_merged_clean_dedup.csv")

    header = _run_cmd_capture(f"head -n 1 '{input_csv}'")
    with open(dedup_path, "w", encoding="utf-8") as fout:
        fout.write(header + "\n")

    # Append unique sorted lines
    cmd = f"tail -n +2 '{input_csv}' | sort -T /tmp -u >> '{dedup_path}'"
    _run_cmd_capture(cmd)

    # Recount for sanity
    out_total_lines_str = _run_cmd_capture(f"wc -l < '{dedup_path}'")
    out_total_lines = int(out_total_lines_str)

    return {
        "dedup_performed": True,
        "input_path": input_csv,
        "output_path": dedup_path,
        "total_lines": total_lines,
        "unique_data_lines": unique_data_lines,
        "duplicates": duplicates,
        "output_total_lines": out_total_lines,
    }


if __name__ == "__main__":
    setup_logging("INFO")

    if not os.path.isdir(data_folder):
        raise SystemExit(f"Data folder not found: {data_folder}")

    # Discover CSV files
    csv_files = [
        os.path.join(data_folder, f)
        for f in sorted(os.listdir(data_folder))
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_folder}")

    logger.info(f"[+] Found {len(csv_files)} CSV files")

    # 1) Detect majority schema and identify outliers
    majority_schema, _ = detect_majority_schema(csv_files)

    # 2) Stream through all files to compute stats while dropping extras from outliers
    num_columns, label_distribution = compute_label_distribution_and_column_count(
        csv_files, majority_schema
    )

    # 3) Merge, clean, and write to destination CSV
    stats = merge_and_clean_to_csv(
        csv_files, majority_schema, dest_merged_folder
    )
    # 3b) Verify and globally de-duplicate if needed
    dedup_stats = verify_and_global_dedupe(stats["output_path"], dest_merged_folder)

    # Results
    logger.info(f"[+] Merged columns count (after alignment): {num_columns}")
    logger.info("[+] Label distribution across merged data:")
    for label, count in label_distribution.items():
        logger.info(f"    {label}: {count}")
    logger.info("[+] Cleaning summary:")
    logger.info(f"    Rows read: {stats['total_read']}")
    logger.info(f"    Rows written: {stats['total_written']}")
    logger.info(f"    Dropped repeated header rows: {stats['dropped_header_rows']}")
    logger.info(f"    Dropped NaN/Inf rows: {stats['dropped_missing_inf']}")
    logger.info(f"    Dropped duplicates (per-chunk): {stats['dropped_duplicates']}")
    logger.info(f"    Output file (pre-dedup): {stats['output_path']}")
    if dedup_stats.get("dedup_performed"):
        logger.info("[+] Global de-duplication performed (disk-based sort -u):")
        logger.info(f"    Total lines (incl. header): {dedup_stats['total_lines']}")
        logger.info(f"    Unique data lines: {dedup_stats['unique_data_lines']}")
        logger.info(f"    Duplicates removed across chunks: {dedup_stats['duplicates']}")
        logger.info(f"    Deduped output file: {dedup_stats['output_path']}")
    else:
        logger.info("[+] No global duplicates detected across chunks.")
        logger.info(f"    Final output file: {dedup_stats['output_path']}")
