import os
import argparse
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018
from src.preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)

# All major labels including Benign/Infilteration
ALL_LABELS = [
    'Benign', 'Infilteration', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 
    'DoS attacks-Hulk', 'Bot', 'SSH-Bruteforce', 'DoS attacks-GoldenEye'
]


def _resolve_coreset_path(subset: str, label: str, filtered_benign: bool = False) -> str:
    """Resolve coreset path based on standardized naming convention."""
    label_safe = cic2018.get_label_name(label)
    
    if subset == 'train':
        # Train: <base>_compressed_coreset.parquet
        if label == 'Benign' and filtered_benign:
            # Benign uses filtered embeddings as base
            filename = f"cic2018_{label_safe}_embedding_filtered_compressed_coreset.parquet"
        else:
            # Other major classes use base embeddings
            filename = f"cic2018_{label_safe}_embedding_compressed_coreset.parquet"
        return os.path.join(cic2018.DATA_FOLDER, 'embeddings', subset, filename)
    else:
        # Test: embeddings/test/cic2018_<Label>_test_selected_rowids.parquet
        filename = f"cic2018_{label_safe}_test_selected_rowids.parquet"
        return os.path.join(cic2018.DATA_FOLDER, 'embeddings', subset, filename)


def _finalize_one(subset: str, label: str, coreset_path: str = None, numerical_inverse: str = 'quantile', filtered_benign: bool = False) -> None:
    """Finalize compressed data for one label."""
    # Resolve coreset path
    if coreset_path is None:
        coreset_path = _resolve_coreset_path(subset, label, filtered_benign=filtered_benign)
    
    if not os.path.exists(coreset_path):
        logger.warning(f"[{label}] Coreset file not found: {coreset_path}")
        return
    
    logger.info(f"[{label}] Loading coreset from: {coreset_path}")
    coreset_df = pd.read_parquet(coreset_path)
    
    if 'RowId' not in coreset_df.columns:
        logger.error(f"[{label}] Coreset must contain RowId column")
        return
    
    row_ids = coreset_df['RowId'].astype(np.int64).tolist()
    logger.info(f"[{label}] Coreset contains {len(row_ids)} RowIds")

    # Load original encoded data
    label_safe = cic2018.get_label_name(label)
    encoded_path = os.path.join(cic2018.ENCODED_DATA_FOLDER, subset, f"cic2018_{label_safe}_encoded.csv")
    
    if not os.path.exists(encoded_path):
        logger.error(f"[{label}] Original encoded file not found: {encoded_path}")
        return
    
    logger.info(f"[{label}] Loading original encoded data from: {encoded_path}")
    encoded_df = pd.read_csv(encoded_path, low_memory=False)
    
    # Map RowIds back to original data
    valid_row_ids = [rid for rid in row_ids if rid < len(encoded_df)]
    if len(valid_row_ids) != len(row_ids):
        logger.warning(f"[{label}] Some RowIds out of range: {len(row_ids) - len(valid_row_ids)} invalid")
    
    # Create subset based on RowIds
    encoded_subset = encoded_df.iloc[valid_row_ids].copy().reset_index(drop=True)
    logger.info(f"[{label}] Extracted {len(encoded_subset)} rows from original encoded data")

    # Save encoded subset
    encoded_out_dir = os.path.join(cic2018.ENCODED_DATA_FOLDER, subset)
    os.makedirs(encoded_out_dir, exist_ok=True)
    encoded_out = os.path.join(encoded_out_dir, f"cic2018_{label_safe}_encoded_compressed.csv")
    
    encoded_subset.to_csv(encoded_out, index=False)
    logger.info(f"[{label}] Saved encoded subset -> {encoded_out}")

    # Decode to raw_processed format
    logger.info(f"[{label}] Decoding to raw_processed format...")
    preprocessor = CIC2018Preprocessor()
    
    try:
        # Load encoders
        preprocessor.load_encoders()
        logger.info(f"[{label}] Loaded pre-fitted encoders")
        
        # Decode the subset
        raw_subset = preprocessor.inverse_transform(encoded_subset, numerical_inverse=numerical_inverse)
        logger.info(f"[{label}] Decoded to raw format: {raw_subset.shape}")
        
        # Save raw_processed subset
        raw_out_dir = os.path.join(cic2018.RAW_PROCESSED_DATA_FOLDER, subset)
        os.makedirs(raw_out_dir, exist_ok=True)
        raw_out = os.path.join(raw_out_dir, f"cic2018_{label_safe}_raw_processed_compressed.csv")
        
        raw_subset.to_csv(raw_out, index=False)
        logger.info(f"[{label}] Saved raw_processed subset -> {raw_out}")
        
    except Exception as e:
        logger.error(f"[{label}] Failed to decode: {e}")
        logger.info(f"[{label}] Encoded subset saved, but decoding failed")

    # Summary
    logger.info(f"[{label}] Finalization complete: {len(encoded_subset)} points")


def main():
    parser = argparse.ArgumentParser(description="Map coreset RowIds back to original encoded and raw_processed data for all major classes")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, default='All', 
                        choices=ALL_LABELS + ['All'])
    parser.add_argument('--coreset-path', type=str, default=None,
                        help='(Ignored in All mode) Path to coreset parquet file')
    parser.add_argument('--benign-source', type=str, default='base', choices=['base', 'filtered'],
                        help='Which Benign embeddings to use: base/original (default) or filtered')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--numerical-inverse', type=str, default='quantile', choices=['quantile', 'minmax'],
                        help='Method to inverse transform numerical features')
    args = parser.parse_args()
    setup_logging(args.log_level)

    # All-mode: iterate labels
    if args.label == 'All':
        for lb in ALL_LABELS:
            try:
                _finalize_one(
                    subset=args.subset,
                    label=lb,
                    coreset_path=None,  # Auto-resolve per label
                    numerical_inverse=args.numerical_inverse,
                    filtered_benign=args.benign_source == 'filtered',
                )
            except Exception as e:
                logger.warning(f"[{lb}] failed: {e}")
        return

    # Single label mode
    _finalize_one(
        subset=args.subset,
        label=args.label,
        coreset_path=args.coreset_path,
        numerical_inverse=args.numerical_inverse,
        filtered_benign=args.benign_source == 'filtered',
    )


if __name__ == "__main__":
    main()
