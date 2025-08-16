import os
import argparse
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from collections import Counter

import configs.cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from dataservice.data_service import DataService
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

data_folder = configs.cic2018.ORIGINAL_DATA_FOLDER
dest_folder = configs.cic2018.DATA_FOLDER


def read_header_columns(csv_path: str) -> List[str]:
    """Read header columns from CSV file"""
    header_df = pd.read_csv(
        csv_path, nrows=0, low_memory=False, skipinitialspace=True
    )
    return [c.strip() for c in header_df.columns.tolist()]


def detect_majority_schema(csv_files: List[str]) -> Tuple[List[str], dict]:
    """
    Detect the majority schema across all CSV files.
    """
    schema_counter: Counter = Counter()
    file_to_schema = {}

    for fp in csv_files:
        cols = read_header_columns(fp)
        file_to_schema[fp] = cols
        schema_counter[tuple(cols)] += 1

    if not schema_counter:
        raise RuntimeError("No CSV headers detected in the provided folder")

    majority_schema_tuple, majority_count = schema_counter.most_common(1)[0]
    majority_schema = list(majority_schema_tuple)

    logger.info(f"[+] Majority schema detected from {majority_count}/{len(csv_files)} files")
    logger.info(f"[+] Majority schema: {len(majority_schema)} columns")
    
    # Check for files with different schemas
    outliers = [fp for fp, cols in file_to_schema.items() if cols != majority_schema]
    if outliers:
        logger.info(f"[+] Found {len(outliers)} files with different schemas:")
        for fp in outliers:
            cols = file_to_schema[fp]
            extra = [c for c in cols if c not in majority_schema]
            missing = [c for c in majority_schema if c not in cols]
            logger.info(f"    - {os.path.basename(fp)}: +{len(extra)} extras, -{len(missing)} missing")
            if extra:
                logger.info(f"      Extra: {extra[:4]}{'...' if len(extra) > 4 else ''}")
    else:
        logger.info(f"[+] All {len(csv_files)} files have identical schema")

    return majority_schema, file_to_schema


def merge_csv_files_with_early_cleaning(
    csv_files: List[str],
    majority_schema: List[str],
    preprocessor: CIC2018Preprocessor,
    chunksize: int = 500_000,
):
    """
    Merge CSV files with early feature selection and cleaning to reduce memory usage.
    """
    logger.info("[+] Phase 1: Merge CSV files with early cleaning...")
    
    merged_chunks = []
    total_read = 0
    dropped_header_rows = 0
    feature_selection_logged = False
    
    for fp in csv_files:
        logger.info(f"[+] Processing: {os.path.basename(fp)}")
        
        for chunk in pd.read_csv(
            fp,
            usecols=majority_schema,  # Use only majority schema columns
            low_memory=False,
            skipinitialspace=True,
            chunksize=chunksize,
        ):
            total_read += len(chunk)
            
            # Early feature selection to reduce memory usage (log only once)
            if not feature_selection_logged:
                logger.info(f"[+] Applying feature selection to reduce memory usage...")
                # Show what columns will be dropped
                columns_to_drop = ['Timestamp', 'Bwd PSH Flags', 'Bwd URG Flags', 
                                 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 
                                 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
                available_drops = [col for col in columns_to_drop if col in chunk.columns]
                if available_drops:
                    logger.info(f"[+] Dropping columns: {available_drops}")
                feature_selection_logged = True
            
            # Temporarily suppress preprocessor logging to avoid spam
            preprocessor_logger = logging.getLogger("preprocessing.cic2018_preprocessor")
            original_level = preprocessor_logger.level
            preprocessor_logger.setLevel(logging.WARNING)
            
            chunk = preprocessor.select_features_and_label(chunk)
            
            # Restore original logging level
            preprocessor_logger.setLevel(original_level)
            
            # Normalize and drop repeated header rows
            if "Label" not in chunk.columns:
                raise RuntimeError(f"Missing 'Label' column in file {fp}")
            
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            before = len(chunk)
            chunk = chunk[chunk["Label"] != "Label"]
            dropped_header_rows += before - len(chunk)
            
            # Early cleaning: remove missing and infinite values to reduce data size
            # Use DataService for consistent cleaning
            chunk = DataService.fix_missing_and_inf_values(chunk)
            chunk = DataService.fix_duplicates(chunk, method="pandas")
            
            if len(chunk) > 0:
                merged_chunks.append(chunk)
    
    logger.info(f"[+] Merging {len(merged_chunks)} chunks...")
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    
    logger.info(f"[+] Phase 1 complete:")
    logger.info(f"    - Total rows read: {total_read}")
    logger.info(f"    - Dropped header rows: {dropped_header_rows}")
    logger.info(f"    - Merged data shape after early cleaning: {merged_df.shape}")
    
    return merged_df


def deduplicate_dataframe(df: pd.DataFrame, stage_name: str = "", precheck: bool = False, dedup_method: str = "auto"):
    """
    Remove duplicates from dataframe using DataService with optimized methods for dataset size.
    """
    logger.info(f"[+] Deduplicating {stage_name} data...")
    
    # Use DataService static methods for consistent deduplication
    has_duplicates = DataService.check_duplicates(df) if precheck else True
    
    if has_duplicates:
        before_dedup = len(df)
        if precheck:
            duplicate_count = df.duplicated().sum()
            logger.info(f"[+] Found {duplicate_count} duplicates in {stage_name} data")
        
        # Use DataService with auto method selection for optimal performance
        df_clean = DataService.fix_duplicates(df, method=dedup_method)
        after_dedup = len(df_clean)
        duplicates_removed = before_dedup - after_dedup
        
        logger.info(f"[+] {stage_name} deduplication: {duplicates_removed} duplicates removed")
        logger.info(f"[+] Final {stage_name} shape: {df_clean.shape}")
        
        # Verify no duplicates remain
        final_dup_check = DataService.check_duplicates(df_clean)
        if final_dup_check:
            remaining_dups = df_clean.duplicated().sum()
            logger.warning(f"[!] WARNING: Still {remaining_dups} duplicates remaining!")
        else:
            logger.info(f"[+] Verification: No duplicates remaining in {stage_name} data")
        
        return df_clean
    else:
        logger.info(f"[+] No duplicates found in {stage_name} data")
        return df


def main():
    """
    Unified preprocessing pipeline:
    1. Merge CSV files with early feature selection and cleaning
    2. Deduplicate merged data
    3. Setup encoders and encode data
    4. Deduplicate encoded data
    5. Export encoded and inverse-transformed data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", action="store_true", help="Precheck for duplicates")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--dedup-method", type=str, default="auto", choices=["auto", "pandas", "streaming", "external"],
                        help="Deduplication method: auto, pandas, streaming, external")
    args = parser.parse_args()
    setup_logging(args.log_level)
    
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
    logger.info("[+] ===========================================")
    logger.info("[+] UNIFIED PREPROCESSING PIPELINE STARTED")
    logger.info("[+] ===========================================")
    
    # Initialize preprocessor
    preprocessor = CIC2018Preprocessor()
    
    # Detect majority schema across all files
    logger.info("[+] Detecting schema across CSV files...")
    majority_schema, file_to_schema = detect_majority_schema(csv_files)
    
    # Phase 1: Merge CSV files with early cleaning
    merged_df = merge_csv_files_with_early_cleaning(csv_files, majority_schema, preprocessor)
    
    # Phase 2: Final deduplication on merged data
    logger.info("[+] Phase 2: Final deduplication on merged data...")
    clean_df = deduplicate_dataframe(merged_df, "merged", dedup_method=args.dedup_method)
    
    logger.info(f"[+] Final clean merged data: {clean_df.shape}")
    logger.info("[+] Label distribution in clean data:")
    for label, count in clean_df['Label'].value_counts().items():
        logger.info(f"    {label}: {count}")
    
    # Phase 3: Setup encoders and encode data
    logger.info("[+] Phase 3: Setting up encoders and encoding...")
    
    # Setup encoders on clean data
    logger.info("[+] Setting up encoders...")
    preprocessor.setup_encoders(clean_df)
    preprocessor.save_encoders()
    
    # Encode data
    logger.info("[+] Encoding features...")
    encoded_df = preprocessor.preprocess_encode_numerical_features(clean_df.copy())
    encoded_df = preprocessor.preprocess_encode_binary_features(encoded_df)
    encoded_df = preprocessor.preprocess_encode_label(encoded_df)
    encoded_df = preprocessor.preprocess_encode_categorical_features(encoded_df)
    
    logger.info(f"[+] Encoded data shape: {encoded_df.shape}")
    logger.info("[+] Encoded label distribution:")
    logger.info(encoded_df['Label'].value_counts())
    
    # Phase 4: Deduplicate encoded data
    logger.info("[+] Phase 4: Deduplicating encoded data...")
    final_encoded_df = deduplicate_dataframe(encoded_df, "encoded", dedup_method=args.dedup_method)
    
    # Phase 5: Export final datasets
    logger.info("[+] Phase 5: Exporting final datasets...")
    
    # Create output directory
    os.makedirs(dest_folder, exist_ok=True)
    
    # Export encoded data
    encoded_output_path = os.path.join(dest_folder, "CIC2018_encoded.csv")
    logger.info(f"[+] Exporting encoded data to: {encoded_output_path}")
    preprocessor.export_encoded_data(final_encoded_df, encoded_output_path)
    
    # Export raw data (inverse transform)
    logger.info("[+] Inverse transforming encoded data...")
    raw_df = preprocessor.inverse_transform(final_encoded_df)
    raw_output_path = os.path.join(dest_folder, "CIC2018_raw_processed.csv")
    logger.info(f"[+] Exporting raw processed data to: {raw_output_path}")
    preprocessor.export_raw_data(raw_df, raw_output_path)
    
    # Final summary
    logger.info("[+] ===========================================")
    logger.info("[+] UNIFIED PREPROCESSING COMPLETED!")
    logger.info("[+] ===========================================")
    logger.info(f"[+] Input files processed: {len(csv_files)}")
    logger.info(f"[+] Final encoded rows: {len(final_encoded_df)}")
    logger.info(f"[+] Final raw processed rows: {len(raw_df)}")
    logger.info(f"[+] Files created:")
    logger.info(f"    - {encoded_output_path} (encoded for WGAN)")
    logger.info(f"    - {raw_output_path} (raw for final training)")
    logger.info(f"[+] Feature breakdown:")
    logger.info(f"    - Categorical (OneHot): {len(preprocessor.encoded_categorical_features)} features")
    logger.info(f"    - Binary: {len(preprocessor.binary_features)} features")
    logger.info(f"    - Numerical (MinMax): {len(preprocessor.encoded_numerical_features)} features")
    logger.info(f"    - Total features after encoding: {final_encoded_df.shape[1] - 1} (excluding Label)")
    logger.info("[+] ===========================================")
    logger.info("[+] Ready for downstream tasks!")


if __name__ == "__main__":
    main()
