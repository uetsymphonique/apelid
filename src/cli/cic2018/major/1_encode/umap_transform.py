import os
import argparse
import pandas as pd
import numpy as np
import joblib

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.embedding import make_metadata_frame, save_embeddings_parquet


logger = get_logger(__name__)
from tqdm import tqdm


ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
EMBED_DIR = os.path.join(cic2018.DATA_FOLDER, "embeddings")


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
    parser = argparse.ArgumentParser(description="Transform encoded data to UMAP embeddings per label (using fitted PCA+UMAP)")
    parser.add_argument('--input-dir', type=str, default=ENCODED_DIR,
                        help='Root directory containing per-label encoded CSVs; choose subset subdir')
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'],
                        help='Subset to transform')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'])
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS)
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=cic2018.MAJORITY_LABELS)
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='PCA cache root (default: configs.cic2018.PCA_CACHE_FOLDER/cache_pca_<subset>/)')
    parser.add_argument('--encoders-dir', type=str, default=cic2018.ENCODERS_FOLDER)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Log hyperparameters
    logger.debug(f"UMAP transform hyperparameters: float32={args.float32}")

    # input_dir is unused in cache-only mode; keep arg for compatibility

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

    # Build label list directly from configs without scanning filenames
    allowed_safe_list = [cic2018.get_label_name(lbl) for lbl in target_labels]

    # Load UMAP model (we always read from PCA cache for transform)
    um = joblib.load(os.path.join(args.encoders_dir, 'umap_major.pkl'))
    umap_dims = int(getattr(um, 'n_components', 0))
    logger.info(f"Loaded UMAP(n={umap_dims})")

    pre = CIC2018Preprocessor()
    pre.load_encoders()

    # Save under subset-specific directory to avoid overwrite
    out_dir = os.path.join(EMBED_DIR, args.subset)
    os.makedirs(out_dir, exist_ok=True)

    for label_safe in tqdm(allowed_safe_list, desc='[umap-transform] per-label'):
        # Always read per-label PCA cache parts and transform using cached PCA outputs
        cache_root = args.cache_dir or os.path.join(cic2018.PCA_CACHE_FOLDER, f'cache_pca_{args.subset}')
        label_cache_dir = os.path.join(cache_root, label_safe)
        if not os.path.isdir(label_cache_dir):
            raise SystemExit(f"PCA cache directory not found for label {label_safe}: {label_cache_dir}")
        part_files = [
            os.path.join(label_cache_dir, f)
            for f in sorted(os.listdir(label_cache_dir))
            if f.endswith('.parquet') and f.startswith('pca_')
        ]
        if not part_files:
            raise SystemExit(f"No PCA cache parts found for label {label_safe} in {label_cache_dir}")
        # Load columns and validate RowId presence
        temp = pd.read_parquet(part_files[0])
        pca_cols = [c for c in temp.columns if c.startswith('pca_')]
        if not pca_cols:
            raise SystemExit("No PCA columns found in cache part")
        if 'RowId' not in temp.columns:
            raise SystemExit("PCA cache parts missing RowId; re-run pca_transform with RowId enabled")
        # Stream parts, transform each block
        Z_blocks: list[np.ndarray] = []
        rowid_blocks: list[pd.Series] = []
        for pf in part_files:
            part_df = pd.read_parquet(pf, columns=pca_cols + ['RowId'])
            if args.float32:
                part_df[pca_cols] = part_df[pca_cols].astype(np.float32, copy=False)
            Z_block = um.transform(part_df[pca_cols].values)
            if args.float32:
                Z_block = Z_block.astype(np.float32, copy=False)
            Z_blocks.append(Z_block)
            rowid_blocks.append(part_df['RowId'].reset_index(drop=True))
        Z = np.vstack(Z_blocks)
        row_id_col = pd.concat(rowid_blocks, ignore_index=True)
        logger.debug(f"UMAP transform {label_safe} from cache: parts={len(part_files)} -> rows={Z.shape[0]} dims={Z.shape[1]}")
        # Since we do not read encoded CSV, set label metadata neutral
        label_id = -1
        label_name = None
        pca_dims = len(pca_cols)

        meta_df = make_metadata_frame(
            n_rows=len(Z),
            label_id=label_id,
            label_name=label_name,
            label_safe=label_safe,
            encoder_num='unknown',
            pca_dims=pca_dims,
            pca_var=float('nan'),
            umap_dims=umap_dims,
            umap_neighbors=-1,
            umap_min_dist=-1.0,
            metric='euclidean',
            seed=42,
            source=f'major-{args.subset}',
        )
        # Append RowId into metadata for robust mapping back to encoded
        meta_df = pd.concat([meta_df.reset_index(drop=True), row_id_col.reset_index(drop=True)], axis=1)
        try:
            rid_min = int(meta_df['RowId'].min())
            rid_max = int(meta_df['RowId'].max())
            logger.debug(f"RowId attached: min={rid_min}, max={rid_max}, rows={len(meta_df)}")
        except Exception:
            logger.info("RowId attached (stats unavailable)")
        out_path = os.path.join(out_dir, f"cic2018_{label_safe}_embedding.parquet")
        save_embeddings_parquet(Z, meta_df, out_path, engine='pyarrow', compression='snappy')
        logger.info(f"Saved embeddings: {out_path} (rows={len(meta_df)}, dims={Z.shape[1]})")


if __name__ == "__main__":
    main()


