import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from ..helpers import load_embeddings
from resampling.undersampling.coreset import CoresetSelector


logger = get_logger(__name__)


DATA_FOLDER = cic2018.DATA_FOLDER
EMBED_DIR = os.path.join(DATA_FOLDER, "embeddings")


from tqdm import tqdm


def _resolve_default_input(subset: str, label: str, benign_source: str = 'filtered') -> str:
    """Single-case input resolution per agreed flow:
    - Benign: use filtered embeddings
    - Infilteration: use base embeddings
    """
    # Use config helper
    use_filtered_benign = (benign_source == 'filtered')
    filtered = (label == 'Benign' and use_filtered_benign)
    path = cic2018.embedding_path(subset, label, filtered_benign=filtered)
    if not os.path.exists(path):
        raise SystemExit(f"Coreset input not found for {label} at expected path: {path}")
    logger.info(f"Using input embeddings: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Balanced coreset selection: uses boundary_score/role in-place on base embeddings + additional reps")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, default='Benign', choices=['Benign', 'Infilteration'])
    parser.add_argument('--input-path', type=str, default=None,
                        help='Base embedding parquet (with boundary_score/role written in-place by compute_boundary)')
    parser.add_argument('--budget-total', type=int, default=14000, help='Final budget (default: 14000)')
    parser.add_argument('--float32', action='store_true', default=True, help='Cast embeddings to float32 (default: enabled)')
    parser.add_argument('--benign-source', type=str, default='base', choices=['base', 'filtered'],
                        help='Which Benign embeddings to use in default input resolution')
    parser.add_argument('--kmeans-batch', type=int, default=10000)
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (default: 1500)')
    parser.add_argument('--out-path', type=str, default=None,
                        help='Output parquet for compressed coreset; default appends _compressed_coreset')
    
    # Additional selection parameters
    parser.add_argument('--overlap-ratio', type=float, default=0.05,
                        help='Ratio of additional budget for overlap (deep-inside-other) points (default: 0.05)')
    parser.add_argument('--min-margin', type=float, default=2.0,
                        help='Minimum margin threshold for core selection (exclude deep-overlap, default: 2.0)')
    parser.add_argument('--use-relative-margin', action='store_true',
                        help='Interpret boundary_score as relative margin (from compute_boundary) for filtering')
    
    # QC options
    parser.add_argument('--qc-cover-sample', type=int, default=300000,
                        help='Max samples for covering radius estimation (nearest selected)')
    parser.add_argument('--qc-enable', action='store_true', default=True, help='Enable QC logs (default: enabled)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # log hyperparameters
    logger.debug(f"Hyperparameters: subset={args.subset}, label={args.label}, budget_total={args.budget_total}, kmeans_batch={args.kmeans_batch}, n_clusters={args.n_clusters}, overlap_ratio={args.overlap_ratio}, min_margin={args.min_margin}, use_relative_margin={args.use_relative_margin}, qc_cover_sample={args.qc_cover_sample}, qc_enable={args.qc_enable}")

    # Validate overlap ratio
    if args.overlap_ratio > 1.0:
        raise SystemExit(f"Overlap ratio ({args.overlap_ratio}) cannot exceed 1.0")
    
    # Resolve input
    if args.input_path is None:
        args.input_path = _resolve_default_input(args.subset, args.label, benign_source=args.benign_source)
    else:
        logger.debug(f"Using user-specified input embeddings: {args.input_path}")

    df = load_embeddings(args.input_path, float32=args.float32)
    logger.info(f"RowId present (input): {'RowId' in df.columns}")
    if 'RowId' not in df.columns:
        raise SystemExit("Input embeddings must contain RowId (ensure umap_transform added it)")

    # Require boundary_score written in-place by compute_boundary
    if 'boundary_score' not in df.columns:
        raise SystemExit("boundary_score not found in embeddings. Run compute_boundary first (now writes in-place).")

    # Identify PIN from role if present
    if 'role' in df.columns:
        pin_mask = (df['role'] == 'boundary')
        pin_idx = np.where(pin_mask)[0]
    else:
        logger.warning("'role' column not found; proceeding with zero PIN and selecting all from non-PIN")
        pin_idx = np.array([], dtype=np.int64)
    logger.info(f"PIN from embeddings: {len(pin_idx)} points")

    z_cols = [c for c in df.columns if c.startswith('z_')]
    if not z_cols:
        raise SystemExit("Embedding columns not found (expected z_1..z_k)")

    X = df[z_cols].to_numpy()

    # Margin values if requested
    margin_values = df['boundary_score'].to_numpy() if args.use_relative_margin else None

    # Use CoresetSelector
    selector = CoresetSelector(kmeans_batch=int(args.kmeans_batch), random_state=42)
    selected_idx, pin_sel, core_idx, overlap_idx = selector.select(
        X=X,
        pin_idx=pin_idx,
        margin_values=margin_values,
        n_clusters=args.n_clusters,
        budget_total=int(args.budget_total),
        overlap_ratio=float(args.overlap_ratio),
        min_margin=float(args.min_margin),
    )

    logger.info(f"Final coreset composition: PIN={len(pin_sel)}, Core={len(core_idx)}, Overlap={len(overlap_idx)}, Total={len(selected_idx)}")

    # Build output with metadata
    out_df = df.iloc[selected_idx].copy().reset_index(drop=True)

    # Assign roles
    role = out_df.get('role', pd.Series(['core'] * len(out_df)))
    if len(pin_sel) > 0:
        role.iloc[:len(pin_sel)] = 'boundary'
    if len(core_idx) > 0:
        role.iloc[len(pin_sel):len(pin_sel)+len(core_idx)] = 'core'
    if len(overlap_idx) > 0:
        role.iloc[len(pin_sel)+len(core_idx):] = 'overlap'
    out_df['role'] = role.values

    # dist_to_S: compute only if needed (pin exists and non-pin selected)
    if len(pin_sel) > 0 and (len(core_idx) > 0 or len(overlap_idx) > 0):
        logger.debug("Computing dist_to_S: distance to nearest PIN")
        X_pin = X[pin_sel]
        X_other = X[np.concatenate([core_idx, overlap_idx])]
        batch = 200000
        dist_other = np.empty(len(X_other), dtype=np.float32)
        for start in tqdm(range(0, len(X_other), batch), desc='dist other->PIN'):
            end = min(start + batch, len(X_other))
            diff = X_other[start:end, None, :] - X_pin[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            dist_other[start:end] = np.sqrt(d2.min(axis=1)).astype(np.float32, copy=False)
        dist_S = np.concatenate([np.zeros(len(pin_sel), dtype=np.float32), dist_other])
    else:
        dist_S = np.zeros(len(selected_idx), dtype=np.float32)
    out_df['dist_to_S'] = dist_S

    # QC logs remain unchanged (optional)
    if args.qc_enable:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
        nn.fit(X[selected_idx])
        population_idx = np.arange(len(df))
        sample_cap = int(args.qc_cover_sample)
        rng = np.random.RandomState(42)
        if len(population_idx) > sample_cap:
            pop_sample = rng.choice(population_idx, size=sample_cap, replace=False)
        else:
            pop_sample = population_idx
        d_cov, _ = nn.kneighbors(X[pop_sample], n_neighbors=1, return_distance=True)
        d_cov = d_cov.ravel().astype(np.float32, copy=False)
        p50 = float(np.percentile(d_cov, 50)); p90 = float(np.percentile(d_cov, 90)); p95 = float(np.percentile(d_cov, 95))
        logger.info(f"Covering radius (nearest selected): p50={p50:.4f}, p90={p90:.4f}, p95={p95:.4f}")
        if 'cluster_id' in df.columns:
            before_counts = df['cluster_id'].value_counts().to_dict()
            after_counts = out_df['cluster_id'].value_counts().to_dict()
        if 'boundary_score' in df.columns:
            bs_series = out_df.loc[out_df['role'] == 'boundary', 'boundary_score'].dropna()
            if len(bs_series) > 0:
                bs = bs_series.to_numpy(); bp50 = float(np.percentile(bs, 50)); bp90 = float(np.percentile(bs, 90)); bp95 = float(np.percentile(bs, 95)); bmean = float(np.mean(bs))
                logger.info(f"boundary_score in coreset [boundary-only]: mean={bmean:.4f}, p50={bp50:.4f}, p90={bp90:.4f}, p95={bp95:.4f}")
            role_counts = out_df['role'].value_counts().to_dict(); logger.info(f"Role distribution: {role_counts}")

    # Output path
    if args.out_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.out_path = f"{base}_compressed_coreset{ext or '.parquet'}"
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out_df.to_parquet(args.out_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"RowId present (output): {'RowId' in out_df.columns}")
    logger.info(f"Saved balanced coreset -> {args.out_path}")


if __name__ == "__main__":
    main()