import os
import argparse
import pandas as pd

import configs.cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

dest_folder = configs.cic2018.DATA_FOLDER
clean_merged_dir = configs.cic2018.CLEAN_MERGED_DATA_FOLDER
encoded_out_dir = configs.cic2018.ENCODED_DATA_FOLDER
raw_out_dir = configs.cic2018.RAW_PROCESSED_DATA_FOLDER


def deduplicate_dataframe(df: pd.DataFrame, stage_name: str = "", precheck: bool = False, dedup_method: str = "auto", postcheck: bool = False, preprocessor: CIC2018Preprocessor = None):
    """
    Remove duplicates from dataframe using DataService with optimized methods for dataset size.
    """
    logger.info(f"[+] Deduplicating {stage_name} data...")
    
    # Use DataService static methods for consistent deduplication
    has_duplicates = preprocessor.check_duplicates(df) if precheck else True
    
    if has_duplicates:
        before_dedup = len(df)
        if precheck:
            duplicate_count = df.duplicated().sum()
            logger.info(f"[+] Found {duplicate_count} duplicates in {stage_name} data")
        
        # Use DataService with auto method selection for optimal performance
        df_clean = preprocessor.fix_duplicates(df, method=dedup_method)
        after_dedup = len(df_clean)
        duplicates_removed = before_dedup - after_dedup
        
        logger.info(f"[+] {stage_name} deduplication: {duplicates_removed} duplicates removed")
        logger.info(f"[+] Final {stage_name} shape: {df_clean.shape}")
        
        if postcheck:
            # Verify no duplicates remain
            final_dup_check = preprocessor.check_duplicates(df_clean)
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
    Phase 3-4 encoding pipeline (per-label):
    1. Load clean merged datasets per label from CLEAN_MERGED_DATA_FOLDER
    2. Setup encoders on the union of all clean merged data
    3. Encode per-label, deduplicate, and export per-label encoded and raw processed files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=clean_merged_dir,
                        help="Input directory containing per-label clean merged CSVs")
    parser.add_argument("--precheck", action="store_true", help="Precheck for duplicates")
    parser.add_argument("--postcheck", action="store_true", help="Postcheck for duplicates")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--dedup-method", type=str, default="auto", choices=["auto", "pandas", "streaming", "external"],
                        help="Deduplication method for encoded data: auto, pandas, streaming, external")
    args = parser.parse_args()
    setup_logging(args.log_level)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    logger.info("[+] ===========================================")
    logger.info("[+] PHASE 3-4 ENCODING PIPELINE (PER-LABEL) STARTED")
    logger.info("[+] ===========================================")
    logger.info(f"[+] Scanning per-label clean merged files in: {input_dir}")

    label_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith('_clean_merged.csv')
    ]
    if not label_files:
        raise SystemExit(f"No per-label clean merged files found in {input_dir}")

    logger.info(f"[+] Found {len(label_files)} label files")

    # Initialize preprocessor and fit encoders on the union of all clean data
    preprocessor = CIC2018Preprocessor()
    logger.info("[+] Phase 3: Setting up encoders across all labels...")

    # Load and concatenate all per-label data to fit encoders (ensure consistent schema)
    union_frames = []
    total_rows = 0
    for lf in label_files:
        df_part = pd.read_csv(lf, low_memory=False)
        union_frames.append(df_part)
        total_rows += len(df_part)
    union_df = pd.concat(union_frames, ignore_index=True)
    logger.info(f"[+] Union clean data for encoder fit: {union_df.shape} (from {len(label_files)} files, {total_rows} rows)")

    preprocessor.setup_encoders(union_df)
    preprocessor.save_encoders()

    # Ensure output directories exist
    os.makedirs(encoded_out_dir, exist_ok=True)
    os.makedirs(raw_out_dir, exist_ok=True)

    # Encode per-label and export
    logger.info("[+] Phase 4: Encoding and exporting per-label datasets...")
    encoded_counts = {}
    raw_counts = {}

    for lf in label_files:
        df_label = pd.read_csv(lf, low_memory=False)
        # Encode
        enc_df = preprocessor.preprocess_encode_numerical_features(df_label.copy())
        enc_df = preprocessor.preprocess_encode_binary_features(enc_df)
        enc_df = preprocessor.preprocess_encode_label(enc_df)
        enc_df = preprocessor.preprocess_encode_categorical_features(enc_df)

        # Dedup encoded per-label
        enc_df = deduplicate_dataframe(enc_df, stage_name=os.path.basename(lf), precheck=args.precheck, dedup_method=args.dedup_method, postcheck=args.postcheck, preprocessor=preprocessor)

        # Determine label-safe name from file name
        base = os.path.basename(lf)
        # base format: cic2018_<label>_clean_merged.csv -> extract <label>
        label_safe = base[len('cic2018_'):-len('_clean_merged.csv')]

        # Export encoded per-label
        enc_path = os.path.join(encoded_out_dir, f"cic2018_{label_safe}_encoded.csv")
        preprocessor.export_encoded_data(enc_df, enc_path)
        encoded_counts[label_safe] = len(enc_df)

        # Inverse transform and export raw processed per-label
        raw_df = preprocessor.inverse_transform(enc_df)
        raw_path = os.path.join(raw_out_dir, f"cic2018_{label_safe}_raw_processed.csv")
        preprocessor.export_raw_data(raw_df, raw_path)
        raw_counts[label_safe] = len(raw_df)

        logger.info(f"[+] {label_safe}: encoded={len(enc_df)}, raw={len(raw_df)}")

    # Final summary
    logger.info("[+] ===========================================")
    logger.info("[+] PHASE 3-4 ENCODING (PER-LABEL) COMPLETED!")
    logger.info("[+] ===========================================")
    logger.info(f"[+] Label files processed: {len(label_files)}")
    logger.info(f"[+] Encoded outputs -> {encoded_out_dir}")
    logger.info(f"[+] Raw processed outputs -> {raw_out_dir}")
    if encoded_counts:
        sample_items = list(encoded_counts.items())[:10]
        logger.info("[+] Sample encoded counts:")
        for name, cnt in sample_items:
            logger.info(f"    {name}: {cnt}")
        if len(encoded_counts) > 10:
            logger.info(f"    ... and {len(encoded_counts) - 10} more labels")
    logger.info("[+] ===========================================")
    logger.info("[+] Ready for downstream tasks!")


if __name__ == "__main__":
    main()
