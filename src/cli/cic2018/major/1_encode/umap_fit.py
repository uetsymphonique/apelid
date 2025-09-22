import os
import argparse
import pandas as pd
import numpy as np
import joblib

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Fit UMAP on cached PCA union parts (train-only)")
    parser.add_argument('--cache-dir', type=str, default=os.path.join(cic2018.PCA_CACHE_FOLDER, 'cache_pca_train'),
                        help='Directory containing PCA parquet parts (from pca_transform.py)')
    parser.add_argument('--umap-components', type=int, default=24)
    parser.add_argument('--umap-n-neighbors', type=int, default=25)
    parser.add_argument('--umap-min-dist', type=float, default=0.25)
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--seed', type=int, default=42)
    # Performance tuning
    parser.add_argument('--n-epochs', type=int, default=200, help='UMAP training epochs (lower is faster)')
    parser.add_argument('--neg-sample-rate', type=int, default=2, help='UMAP negative_sample_rate (lower is faster)')
    parser.add_argument('--learning-rate', type=float, default=1.0, help='UMAP learning rate')
    parser.add_argument('--low-memory', action='store_true', help='Enable UMAP low_memory mode')
    parser.add_argument('--fit-mode', type=str, default='cap', choices=['cap', 'full'],
                        help='cap: Benign <= 200k, others <= 100k (streaming by label); full: fit on all rows')
    parser.add_argument('--numba-threads', type=int, default=None, help='Limit NUMBA_NUM_THREADS during UMAP fit')
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--encoders-dir', type=str, default=cic2018.ENCODERS_FOLDER)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Log hyperparameters
    logger.info(f"UMAP fit hyperparameters: mode={args.fit_mode}, n_components={args.umap_components}, n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}, metric={args.metric}, seed={args.seed}, n_epochs={args.n_epochs}, neg_sample_rate={args.neg_sample_rate}, learning_rate={args.learning_rate}, low_memory={args.low_memory}, numba_threads={args.numba_threads}, float32={args.float32}")


    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"Cache dir not found: {args.cache_dir}")

    # Detect per-label subdirectories for streaming, class-capped sampling
    label_dirs = [d for d in sorted(os.listdir(args.cache_dir)) if os.path.isdir(os.path.join(args.cache_dir, d))]
    Z_union = None
    pca_cols: list[str] | None = None
    if args.fit_mode == 'cap':
        collected_counts: dict[str, int] = {}
        if label_dirs:
            logger.info("Streaming class-capped sampling from per-label PCA cache subdirectories")
            for label_safe in label_dirs:
                dir_path = os.path.join(args.cache_dir, label_safe)
                part_files = [
                    os.path.join(dir_path, f)
                    for f in sorted(os.listdir(dir_path))
                    if f.endswith('.parquet') and f.startswith('pca_')
                ]
                if not part_files:
                    continue
                # Determine PCA columns once
                if pca_cols is None:
                    temp_df = pd.read_parquet(part_files[0])
                    pca_cols = [c for c in temp_df.columns if c.startswith('pca_')]
                    if not pca_cols:
                        raise SystemExit("No PCA columns found in cache parts")
                cap = 200_000 if label_safe == 'benign' else 100_000
                taken = 0
                sampled_frames: list[pd.DataFrame] = []
                for pf in part_files:
                    if taken >= cap:
                        break
                    df = pd.read_parquet(pf, columns=pca_cols)
                    need = cap - taken
                    if len(df) <= need:
                        sampled_frames.append(df)
                        taken += len(df)
                    else:
                        sampled_frames.append(df.sample(n=need, random_state=args.seed))
                        taken += need
                if not sampled_frames:
                    continue
                label_block = pd.concat(sampled_frames, ignore_index=True)
                collected_counts[label_safe] = len(label_block)
                if Z_union is None:
                    Z_union = label_block
                else:
                    Z_union = pd.concat([Z_union, label_block], ignore_index=True)
            if Z_union is None:
                raise SystemExit(f"No PCA parquet parts found in {args.cache_dir}")
            logger.info(f"After class-capped streaming sampling: rows={len(Z_union)}, per-label-safe={collected_counts}")
        else:
            # Flat cache structure; require Label column for cap
            part_files: list[str] = []
            for root, _, files in os.walk(args.cache_dir):
                for f in files:
                    if f.endswith('.parquet') and f.startswith('pca_'):
                        part_files.append(os.path.join(root, f))
            if not part_files:
                raise SystemExit(f"No PCA parquet parts found in {args.cache_dir}")
            frames = [pd.read_parquet(pf) for pf in tqdm(sorted(part_files), desc='load parts')]
            Z_union = pd.concat(frames, ignore_index=True)
            label_col = cic2018.LABEL_COLUMN
            if label_col not in Z_union.columns:
                raise SystemExit("Flat PCA cache does not contain Label column; re-run pca_transform or use per-label cache")
            benign_cap = 200_000
            other_cap = 100_000
            vc = Z_union[label_col].value_counts()
            logger.info(f"Label counts before sampling: {vc.to_dict()}")
            def _cap_sample(group: pd.DataFrame) -> pd.DataFrame:
                cap = benign_cap if group.name == 'Benign' else other_cap
                n = min(len(group), cap)
                if len(group) <= n:
                    return group
                return group.sample(n=n, random_state=args.seed)
            Z_union = Z_union.groupby(label_col, group_keys=False).apply(_cap_sample)
            vc_after = Z_union[label_col].value_counts()
            logger.info(f"After class-capped sampling (Benign≤{benign_cap}, others≤{other_cap}): rows={len(Z_union)}, per-label={vc_after.to_dict()}")
    else:  # fit_mode == 'full'
        part_files: list[str] = []
        for root, _, files in os.walk(args.cache_dir):
            for f in files:
                if f.endswith('.parquet') and f.startswith('pca_'):
                    part_files.append(os.path.join(root, f))
        if not part_files:
            raise SystemExit(f"No PCA parquet parts found in {args.cache_dir}")
        frames = [pd.read_parquet(pf) for pf in tqdm(sorted(part_files), desc='load parts')]
        Z_union = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded PCA union (full): rows={len(Z_union)}, cols={Z_union.shape[1]}")
    # Ensure numeric dtype post-assembly if requested
    if args.float32:
        if pca_cols is None:
            pca_cols = [c for c in Z_union.columns if c.startswith('pca_')]
        Z_union[pca_cols] = Z_union[pca_cols].astype(np.float32, copy=False)

    # Optional thread limiting for numba
    if args.numba_threads is not None and args.numba_threads > 0:
        os.environ['NUMBA_NUM_THREADS'] = str(int(args.numba_threads))

    import umap
    um = umap.UMAP(
        n_components=int(args.umap_components),
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        metric=args.metric,
        random_state=int(args.seed),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        negative_sample_rate=int(args.neg_sample_rate),
        low_memory=bool(args.low_memory),
    )
    logger.info(f"Fitting UMAP: n_components={args.umap_components}, n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}, metric={args.metric}")
    # Drop label column before fitting (if present)
    X_fit = Z_union.drop(columns=[cic2018.LABEL_COLUMN], errors='ignore').values
    um.fit(X_fit)

    os.makedirs(args.encoders_dir, exist_ok=True)
    joblib.dump(um, os.path.join(args.encoders_dir, 'umap_major.pkl'))
    logger.info(f"Saved UMAP -> {os.path.join(args.encoders_dir, 'umap_major.pkl')}")


if __name__ == "__main__":
    main()


