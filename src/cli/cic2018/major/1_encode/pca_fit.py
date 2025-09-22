import os
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)
from tqdm import tqdm  # type: ignore




ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER


def _select_label_files(input_dir: str, allowed_safe_labels: set[str], subset: str = 'train') -> list[str]:
    files: list[str] = []
    scan_dir = os.path.join(input_dir, subset)
    if not os.path.isdir(scan_dir):
        return files
    for fname in sorted(os.listdir(scan_dir)):
        if not fname.endswith('_encoded.csv'):
            continue
        base = fname
        if not base.startswith('cic2018_'):
            continue
        label_safe = base[len('cic2018_'):-len('_encoded.csv')]
        if label_safe in allowed_safe_labels:
            files.append(os.path.join(scan_dir, fname))
    return files


def main():
    parser = argparse.ArgumentParser(description="Fit PCA (IncrementalPCA recommended) on union of MAJORITY encoded TRAIN data (avoid leakage)")
    parser.add_argument('--input-dir', type=str, default=ENCODED_DIR,
                        help='Root directory containing per-label encoded CSVs; TRAIN subset is used (input-dir/train)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Use all majority labels or a provided list')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS,
                        help='List of label names to include when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS,
                        help='Labels to exclude')
    parser.add_argument('--subset', type=str, default='train', choices=['train'],
                        help='Subset to fit on (fixed to train to avoid leakage)')

    parser.add_argument('--pca-components', type=int, default=32, help='Number of PCA components')
    parser.add_argument('--ipca-batch-size', type=int, default=100000, help='IncrementalPCA batch size')
    parser.add_argument('--read-chunk-rows', type=int, default=200000, help='Rows per chunk when streaming encoded CSVs')
    parser.add_argument('--float32', action='store_true', help='Cast feature matrix to float32 before PCA')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Log hyperparameters
    logger.debug(f"PCA fit hyperparameters: pca_components={args.pca_components}, ipca_batch_size={args.ipca_batch_size}, read_chunk_rows={args.read_chunk_rows}, float32={args.float32}")

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Resolve labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        target_labels = args.labels
    else:
        target_labels = cic2018.MAJORITY_LABELS
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        target_labels = [lbl for lbl in target_labels if lbl not in excludes]
        logger.info(f"Excluding labels: {sorted(excludes)}")
    if not target_labels:
        raise SystemExit("No labels to fit after applying excludes.")
    logger.info(f"Target labels: {target_labels}")

    allowed_safe = {cic2018.get_label_name(lbl) for lbl in target_labels}
    label_files = _select_label_files(input_dir, allowed_safe, subset=args.subset)
    if not label_files:
        raise SystemExit(f"No per-label encoded TRAIN files found in {os.path.join(input_dir, args.subset)}")
    logger.info(f"Found {len(label_files)} encoded label files (subset={args.subset})")

    pre = CIC2018Preprocessor()
    pre.load_encoders()

    def _iter_feature_chunks(files: list[str], chunk_rows: int):
        for fp in files:
            try:
                for chunk in pd.read_csv(fp, low_memory=False, chunksize=chunk_rows):
                    X = chunk.drop(columns=['Label'], errors='ignore')
                    # Drop any *_is_missing indicator columns from features
                    drop_cols = [c for c in X.columns if c.endswith('_is_missing')]
                    if drop_cols:
                        X = X.drop(columns=drop_cols)
                    if args.float32:
                        X = X.astype(np.float32, copy=False)
                    yield X
            except ValueError:
                df = pd.read_csv(fp, low_memory=False)
                X = df.drop(columns=['Label'], errors='ignore')
                drop_cols = [c for c in X.columns if c.endswith('_is_missing')]
                if drop_cols:
                    X = X.drop(columns=drop_cols)
                if args.float32:
                    X = X.astype(np.float32, copy=False)
                yield X

    ipca = IncrementalPCA(n_components=int(args.pca_components), batch_size=int(args.ipca_batch_size))
    total_rows = 0
    feature_dim = None
    rows_bar = tqdm(total=None, desc="PCA fit rows", unit="rows")
    for X in _iter_feature_chunks(label_files, args.read_chunk_rows):
        total_rows += len(X)
        rows_bar.update(len(X))
        if feature_dim is None:
            feature_dim = X.shape[1]
            logger.debug(f"First chunk shape: rows={len(X)}, cols={feature_dim} (float32={bool(args.float32)})")
        ipca.partial_fit(X)
    try:
        rows_bar.close()
    except Exception:
        pass

    os.makedirs(pre.encoders_dir, exist_ok=True)
    joblib.dump(ipca, os.path.join(pre.encoders_dir, 'pca_major.pkl'))
    logger.info(f"IncrementalPCA saved -> {os.path.join(pre.encoders_dir, 'pca_major.pkl')} (rows={total_rows}, input_cols={feature_dim}, n_components={ipca.n_components})")


if __name__ == "__main__":
    main()


