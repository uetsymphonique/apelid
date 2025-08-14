import os
from typing import Dict, Set

import pandas as pd
import configs.cic2018
from dataservice.extractor import Extractor 
from utils.logging import setup_logging, get_logger


DEST_FOLDER = configs.cic2018.DATA_FOLDER
OUTPUT_FILENAME = "CIC2018_merged_clean_dedup.csv"  # prefer deduped if exists


logger = get_logger(__name__)


def resolve_input_path() -> str:
    dedup_path = os.path.join(DEST_FOLDER, "CIC2018_merged_clean_dedup.csv")
    clean_path = os.path.join(DEST_FOLDER, "CIC2018_merged_clean.csv")
    raw_path = os.path.join(DEST_FOLDER, "CIC2018_raw_processed.csv")
    if os.path.exists(raw_path):
        return raw_path
    if os.path.exists(dedup_path):
        return dedup_path
    if os.path.exists(clean_path):
        return clean_path
    raise SystemExit(f"No cleaned CSV found: {dedup_path} or {clean_path} or {raw_path}")


def detect_binary_and_constant_columns(csv_path: str, chunksize: int = 500_000):
    # Read columns
    header_df = pd.read_csv(csv_path, nrows=0, low_memory=False)
    columns = header_df.columns.tolist()

    # Focus on suspected binary columns
    # suspected_binary = [
    #     'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    #     'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    #     'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    #     'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
    #     'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'
    # ]
    
    suspected_binary = ['Protocol', 'Dst Port']

    return Extractor.unique_values(csv_path, chunk_size=chunksize, columns=suspected_binary)


if __name__ == "__main__":
    setup_logging("INFO")
    csv_path = resolve_input_path()
    logger.info(f"[+] Inspecting suspected binary columns from: {csv_path}")

    binary_cols = detect_binary_and_constant_columns(csv_path)

    logger.info("[+] Unique values in suspected binary columns:")
    for col, vals in binary_cols.items():
        logger.info(f"    {col}: {vals} (count: {len(vals)})")

    logger.info(f"[+] Summary: {len(binary_cols)} columns inspected")


