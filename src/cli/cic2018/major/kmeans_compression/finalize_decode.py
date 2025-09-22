import os
import argparse
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


MAJOR_LABELS = [
    'Benign',
    'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
    'Infilteration', 'SSH-Bruteforce', 
    # 'DoS attacks-GoldenEye', 'DoS attacks-Hulk'
]


def _resolve_pca_snapshot_path(subset: str, label: str) -> str:
    """Resolve PCA snapshot path produced by kmeans_compression.

    Train: embeddings/train/pca_cic2018_<label_safe>_compressed_coreset.parquet
    Test:  embeddings/test/pca_cic2018_<label_safe>_test_selected_rowids.parquet
    """
    label_safe = cic2018.get_label_name(label)
    base_dir = os.path.join(cic2018.EMBEDDINGS_FOLDER, subset)
    if subset == 'train':
        filename = f"pca_cic2018_{label_safe}_compressed_coreset.parquet"
    else:
        filename = f"pca_cic2018_{label_safe}_test_selected_rowids.parquet"
    return os.path.join(base_dir, filename)


def _finalize_one(subset: str, label: str, snapshot_path: str | None, numerical_inverse: str) -> None:
    # Resolve snapshot path
    if snapshot_path is None:
        snapshot_path = _resolve_pca_snapshot_path(subset, label)

    if not os.path.exists(snapshot_path):
        logger.warning(f"[{label}] PCA snapshot not found: {snapshot_path}")
        return

    logger.info(f"[{label}] Loading PCA snapshot: {snapshot_path}")
    snap_df = pd.read_parquet(snapshot_path)

    if 'RowId' not in snap_df.columns:
        logger.error(f"[{label}] Snapshot must contain RowId column")
        return

    row_ids = snap_df['RowId'].astype(np.int64).tolist()
    logger.info(f"[{label}] Snapshot contains {len(row_ids)} RowIds")

    # Load original per-label encoded CSV
    label_safe = cic2018.get_label_name(label)
    encoded_path = os.path.join(cic2018.ENCODED_DATA_FOLDER, subset, f"cic2018_{label_safe}_encoded.csv")
    if not os.path.exists(encoded_path):
        logger.error(f"[{label}] Encoded file not found: {encoded_path}")
        return

    logger.info(f"[{label}] Loading encoded data: {encoded_path}")
    encoded_df = pd.read_csv(encoded_path, low_memory=False)

    # Map RowIds (RowId is 0..N-1 within the per-label encoded CSV)
    valid_row_ids = [rid for rid in row_ids if 0 <= rid < len(encoded_df)]
    if len(valid_row_ids) != len(row_ids):
        logger.warning(f"[{label}] {len(row_ids) - len(valid_row_ids)} RowIds out of range; they will be ignored")

    encoded_subset = encoded_df.iloc[valid_row_ids].copy().reset_index(drop=True)
    logger.info(f"[{label}] Extracted {len(encoded_subset)} encoded rows")

    # Save encoded subset
    enc_out_dir = os.path.join(cic2018.ENCODED_DATA_FOLDER, subset)
    os.makedirs(enc_out_dir, exist_ok=True)
    # Align naming with 5_complete (no PCA prefix at this stage)
    enc_out = os.path.join(enc_out_dir, f"cic2018_{label_safe}_encoded_compressed.csv")
    encoded_subset.to_csv(enc_out, index=False)
    logger.info(f"[{label}] Saved encoded subset -> {enc_out}")

    # Decode to raw_processed
    logger.info(f"[{label}] Decoding encoded subset to raw_processed...")
    preprocessor = CIC2018Preprocessor()
    try:
        preprocessor.load_encoders()
        raw_subset = preprocessor.inverse_transform(encoded_subset, numerical_inverse=numerical_inverse)
        raw_out_dir = os.path.join(cic2018.RAW_PROCESSED_DATA_FOLDER, subset)
        os.makedirs(raw_out_dir, exist_ok=True)
        # Align naming with 5_complete (no PCA prefix at this stage)
        raw_out = os.path.join(raw_out_dir, f"cic2018_{label_safe}_raw_processed_compressed.csv")
        raw_subset.to_csv(raw_out, index=False)
        logger.info(f"[{label}] Saved raw_processed subset -> {raw_out}")
    except Exception as e:
        logger.error(f"[{label}] Failed to decode: {e}")
        logger.info(f"[{label}] Encoded subset saved, but decoding failed")

    logger.info(f"[{label}] PCA finalization complete: {len(encoded_subset)} rows")


def main():
    parser = argparse.ArgumentParser(description="Finalize PCA KMeans compression: map RowIds back to encoded and decode to raw_processed")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, default='All')
    parser.add_argument('--snapshot-path', type=str, default=None, help='Override PCA snapshot parquet path for single label')
    parser.add_argument('--numerical-inverse', type=str, default='quantile_normal', choices=['quantile_normal', 'quantile_uniform', 'minmax'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.label == 'All':
        for lb in MAJOR_LABELS:
            try:
                _finalize_one(
                    subset=args.subset,
                    label=lb,
                    snapshot_path=None,
                    numerical_inverse=args.numerical_inverse,
                )
            except Exception as e:
                logger.warning(f"[{lb}] failed: {e}")
        return

    _finalize_one(
        subset=args.subset,
        label=args.label,
        snapshot_path=args.snapshot_path,
        numerical_inverse=args.numerical_inverse,
    )


if __name__ == "__main__":
    main()


