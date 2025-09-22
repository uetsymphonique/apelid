import os
import argparse
import pandas as pd
import numpy as np
import joblib
import shutil

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER


def _select_label_files(input_dir: str, allowed_safe_labels: set[str], subset: str) -> list[str]:
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
    parser = argparse.ArgumentParser(description="Transform encoded data to PCA space per chunk and cache parquet parts")
    parser.add_argument('--input-dir', type=str, default=ENCODED_DIR,
                        help='Root directory containing per-label encoded CSVs; choose subset subdir')
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'],
                        help='Subset to transform')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'])
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS)
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS)
    parser.add_argument('--read-chunk-rows', type=int, default=200000)
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--cache-dir', type=str, default=None,
                        help=f'Directory to save PCA parquet parts; default {cic2018.PCA_CACHE_FOLDER}/cache_pca_<subset>/')
    parser.add_argument('--clean-cache', action='store_true', help='Remove existing PCA cache for target labels before writing')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Log hyperparameters
    logger.debug(f"PCA transform hyperparameters: read_chunk_rows={args.read_chunk_rows}, float32={args.float32}")

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        target_labels = args.labels
    else:
        target_labels = cic2018.MAJORITY_LABELS
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        target_labels = [lbl for lbl in target_labels if lbl not in excludes]
    if not target_labels:
        raise SystemExit("No labels to transform after applying excludes.")

    allowed_safe = {cic2018.get_label_name(lbl) for lbl in target_labels}
    files = _select_label_files(input_dir, allowed_safe, subset=args.subset)
    if not files:
        raise SystemExit(f"No per-label encoded {args.subset} files found in {os.path.join(input_dir, args.subset)}")

    # Load PCA model
    pca = joblib.load(os.path.join(cic2018.ENCODERS_FOLDER, 'pca_major.pkl'))
    n_comp = int(getattr(pca, 'n_components_', getattr(pca, 'n_components', 0)))
    logger.info(f"Loaded PCA model: n_components={n_comp}")

    cache_dir = args.cache_dir or os.path.join(cic2018.PCA_CACHE_FOLDER, f'cache_pca_{args.subset}')
    os.makedirs(cache_dir, exist_ok=True)

    part_idx_by_label: dict[str, int] = {}
    for fp in files:
        base = os.path.basename(fp)
        # Expect: cic2018_<label_safe>_encoded.csv
        label_safe = base[len('cic2018_'):-len('_encoded.csv')] if base.startswith('cic2018_') and base.endswith('_encoded.csv') else 'unknown'
        label_cache_dir = os.path.join(cache_dir, label_safe)
        # Clean old cache if requested
        if args.clean_cache and os.path.isdir(label_cache_dir):
            try:
                logger.info(f"Cleaning existing PCA cache for {label_safe}: {label_cache_dir}")
                shutil.rmtree(label_cache_dir)
            except Exception as e:
                logger.warning(f"Failed to remove cache dir {label_cache_dir}: {e}")
        os.makedirs(label_cache_dir, exist_ok=True)
        try:
            # Track RowId relative to the per-label encoded CSV file
            row_offset = 0
            for chunk in pd.read_csv(fp, low_memory=False, chunksize=args.read_chunk_rows):
                X = chunk.drop(columns=['Label'], errors='ignore')
                drop_cols = [c for c in X.columns if c.endswith('_is_missing')]
                if drop_cols:
                    X = X.drop(columns=drop_cols)
                if args.float32:
                    X = X.astype(np.float32, copy=False)
                Zp = pca.transform(X)
                if args.float32:
                    Zp = Zp.astype(np.float32, copy=False)
                part_idx = part_idx_by_label.get(label_safe, 0)
                out_name = f"pca_{args.subset}_{label_safe}_part_{part_idx:05d}.parquet"
                logger.debug(f"PCA transform part {label_safe}:{part_idx}: X.shape={X.shape} -> Zp.shape={Zp.shape}")
                # Compute RowId for this chunk relative to the source encoded CSV (per label)
                row_ids = np.arange(len(chunk), dtype=np.int64) + row_offset
                df_out = pd.DataFrame(Zp, columns=[f'pca_{i+1}' for i in range(Zp.shape[1])])
                df_out['RowId'] = row_ids
                df_out.to_parquet(
                    os.path.join(label_cache_dir, out_name), engine='pyarrow', compression='snappy', index=False
                )
                part_idx_by_label[label_safe] = part_idx + 1
                row_offset += len(chunk)
        except ValueError:
            df = pd.read_csv(fp, low_memory=False)
            X = df.drop(columns=['Label'], errors='ignore')
            drop_cols = [c for c in X.columns if c.endswith('_is_missing')]
            if drop_cols:
                X = X.drop(columns=drop_cols)
            if args.float32:
                X = X.astype(np.float32, copy=False)
            Zp = pca.transform(X)
            if args.float32:
                Zp = Zp.astype(np.float32, copy=False)
            part_idx = part_idx_by_label.get(label_safe, 0)
            out_name = f"pca_{args.subset}_{label_safe}_part_{part_idx:05d}.parquet"
            # Whole-file fallback; RowId is simply 0..N-1 for the per-label file
            row_ids = np.arange(len(df), dtype=np.int64)
            df_out = pd.DataFrame(Zp, columns=[f'pca_{i+1}' for i in range(Zp.shape[1])])
            df_out['RowId'] = row_ids
            df_out.to_parquet(
                os.path.join(label_cache_dir, out_name), engine='pyarrow', compression='snappy', index=False
            )
            part_idx_by_label[label_safe] = part_idx + 1

    total_parts = sum(part_idx_by_label.values())
    logger.info(f"Wrote {total_parts} PCA parts to {cache_dir} (per-label parts: {part_idx_by_label})")


if __name__ == "__main__":
    main()


