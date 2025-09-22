import os
import argparse
import numpy as np
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018
from resampling.undersampling import DensityAwareFilter


logger = get_logger(__name__)

DATA_FOLDER = cic2018.DATA_FOLDER
EMBED_DIR = os.path.join(DATA_FOLDER, "embeddings")

def filter_benign_embeddings(emb_df: pd.DataFrame,
                             kmeans_k: int,
                             kmeans_batch: int,
                             small_cluster_frac: float,
                             edge_percentile: float,
                             large_keep_rate: float,
                             min_keep_per_large: int,
                             seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    daf = DensityAwareFilter(
        kmeans_k=kmeans_k,
        kmeans_batch=kmeans_batch,
        small_cluster_frac=small_cluster_frac,
        edge_percentile=edge_percentile,
        large_keep_rate=large_keep_rate,
        min_keep_per_large=min_keep_per_large,
        random_state=seed,
    )
    # Follow class pipeline manually to expose centers
    z_cols = [c for c in emb_df.columns if c.startswith('z_')]
    if not z_cols:
        raise SystemExit("Embedding columns not found (expected z_1..z_k)")
    Z = emb_df[z_cols].to_numpy(dtype=np.float32, copy=False)
    labels, centers = daf.fit_microclusters(Z)
    tmp_df = emb_df.copy()
    tmp_df['cluster_id'] = labels
    dist = daf._distance_to_centroid(Z, labels, {int(i): ctr for i, ctr in enumerate(centers)}) if centers is not None else np.zeros(len(Z), dtype=np.float32)
    keep = daf._density_aware_keep_mask(
        labels=labels,
        distances=dist,
        total_rows=len(Z),
        outlier_label=-1,
    )
    filtered = tmp_df[keep].reset_index(drop=True)
    return filtered, centers


def main():
    parser = argparse.ArgumentParser(description="Density-aware filter for Benign embeddings (train-only)")
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to Benign embedding parquet (default: embeddings/train/cic2018_benign_embedding.parquet)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output parquet for filtered Benign embeddings; default appends _filtered')
    parser.add_argument('--kmeans-k', type=int, default=1500, help='MiniBatchKMeans number of clusters')
    parser.add_argument('--kmeans-batch', type=int, default=10000, help='MiniBatchKMeans batch size')
    default_centers_path = cic2018.kmeans_centers_path_train('Benign')
    parser.add_argument('--save-centers', type=str, default=default_centers_path,
                        help=f'Path to save KMeans centers (train); default: {default_centers_path}')
    parser.add_argument('--small-cluster-frac', type=float, default=0.0004,
                        help='Clusters with size <= frac * total are kept entirely (e.g., 0.001 = 0.1%%)')
    parser.add_argument('--edge-percentile', type=float, default=95.0, help='Keep points beyond this distance percentile in each cluster')
    parser.add_argument('--large-keep-rate', type=float, default=0.015, help='Downsample rate for large clusters (e.g., 0.02..0.05)')
    parser.add_argument('--min-keep-per-large', type=int, default=50, help='Minimum kept per large cluster')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)


    # Log hyperparameters
    logger.debug(f"Density-aware filter hyperparameters: kmeans_k={args.kmeans_k}, \
    kmeans_batch={args.kmeans_batch}, small_cluster_frac={args.small_cluster_frac}, \
    edge_percentile={args.edge_percentile}, large_keep_rate={args.large_keep_rate}, \
    min_keep_per_large={args.min_keep_per_large}, seed={args.seed}")

    # Resolve input/output paths
    if args.input_path is None:
        benign_safe = cic2018.get_label_name('Benign')
        train_dir = os.path.join(EMBED_DIR, 'train')
        args.input_path = os.path.join(train_dir, f"cic2018_{benign_safe}_embedding.parquet")
    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_filtered{ext or '.parquet'}"

    if not os.path.exists(args.input_path):
        raise SystemExit(f"Input embedding parquet not found: {args.input_path}")
    

    logger.info(f"Loading embeddings from {args.input_path}")
    emb_df = pd.read_parquet(args.input_path)
    # Check RowId presence
    has_rowid = 'RowId' in emb_df.columns
    logger.info(f"RowId present in input: {has_rowid}")

    logger.info(f"Filtering {len(emb_df)} benign embeddings")
    filtered, centers = filter_benign_embeddings(
        emb_df=emb_df,
        kmeans_k=args.kmeans_k,
        kmeans_batch=args.kmeans_batch,
        small_cluster_frac=args.small_cluster_frac,
        edge_percentile=args.edge_percentile,
        large_keep_rate=args.large_keep_rate,
        min_keep_per_large=args.min_keep_per_large,
        seed=args.seed,
    )

    # Final kept/total summary
    kept, total = len(filtered), len(emb_df)
    ratio = kept / max(1, total)
    logger.info(f"Final kept after density filter: {kept} / {total} ({ratio:.2%})")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    filtered.to_parquet(args.output_path, engine='pyarrow', compression='snappy', index=False)
    # Verify RowId pass-through
    try:
        kept_rowid = 'RowId' in filtered.columns
        logger.info(f"RowId present in output: {kept_rowid}")
    except Exception:
        pass
    # Save centers if requested
    if args.save_centers:
        os.makedirs(os.path.dirname(args.save_centers), exist_ok=True)
        # centers may be a dict (id -> centroid). Convert to ndarray sorted by key.
        if isinstance(centers, dict):
            keys = sorted(centers.keys())
            centers_arr = np.stack([centers[k] for k in keys], axis=0)
        else:
            centers_arr = centers
        np.save(args.save_centers, centers_arr)
        logger.info(f"Saved KMeans centers -> {args.save_centers} (shape={centers_arr.shape})")
    logger.info(f"Saved filtered Benign embeddings -> {args.output_path}")


if __name__ == "__main__":
    main()


