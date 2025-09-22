import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from resampling.undersampling.boundary import BoundarySelector


logger = get_logger(__name__)


def _load_embeddings(path: str, float32: bool) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Embedding parquet not found: {path}")
    df = pd.read_parquet(path)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    if not z_cols:
        raise SystemExit(f"No embedding columns (z_*) in: {path}")
    if float32:
        df[z_cols] = df[z_cols].astype(np.float32, copy=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute boundary scores between target label and Benign filtered embeddings (kNN-based). Supports reusing existing scores to re-select with a new budget.")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--target', type=str, default='Infilteration', choices=['Infilteration', 'Benign'])
    parser.add_argument('--neighbor', type=str, default='Benign', choices=['Infilteration', 'Benign'],
                        help='Neighbor label to compute boundary against (Benign side will always use filtered embeddings)')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100000)
    parser.add_argument('--score', type=str, default='min', choices=['min', 'mean', 'p95'])
    parser.add_argument('--use-relative-margin', action='store_true', 
                        help='Use relative margin (d_cross/d_same) instead of raw cross-class distance')
    parser.add_argument('--margin-low', type=float, default=0.5, 
                        help='Lower threshold for relative margin band (exclude deep-inside-other)')
    parser.add_argument('--margin-high', type=float, default=2.0, 
                        help='Upper threshold for relative margin band (exclude deep-inside-own)')
    parser.add_argument('--select', action='store_true', help='Select top-N boundary points (closest to neighbor) by score')
    parser.add_argument('--budget-total', type=int, default=7000)
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--reuse-existing', action='store_true',
                        help="Reuse existing boundary_score in target parquet and only perform selection with new budget/params (skip recompute)")
    parser.add_argument('--benign-source', type=str, default='base', choices=['base', 'filtered'],
                        help='Which Benign embeddings to use: base (default) or filtered')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Log effective hyperparameters
    logger.debug(f"Boundary params: subset={args.subset}, target={args.target}, neighbor={args.neighbor}, k={args.k}, batch_size={args.batch_size}, score={args.score}, use_relative_margin={args.use_relative_margin}, margin_band=[{args.margin_low}, {args.margin_high}], select={args.select}, budget={args.budget_total}, float32={args.float32}")

    # Resolve paths (Benign side always uses filtered embeddings)
    use_filtered_benign = (args.benign_source == 'filtered')
    target_filtered = (args.target == 'Benign' and use_filtered_benign)
    neighbor_filtered = (args.neighbor == 'Benign' and use_filtered_benign)
    target_path = cic2018.embedding_path(args.subset, args.target, filtered_benign=target_filtered)
    neighbor_path = cic2018.embedding_path(args.subset, args.neighbor, filtered_benign=neighbor_filtered)
    logger.info(f"Target ({args.target}{' filtered' if target_filtered else ''}): {target_path}")
    logger.info(f"Neighbor ({args.neighbor}{' filtered' if neighbor_filtered else ''}): {neighbor_path}")

    # Load target embeddings first (needed for reuse)
    df_target = _load_embeddings(target_path, float32=args.float32)

    # Fast path: reuse existing boundary_score to re-select only
    if args.reuse_existing:
        if 'boundary_score' not in df_target.columns:
            raise SystemExit("reuse-existing requested but 'boundary_score' not found in target parquet. Run compute without --reuse-existing first.")
        if not args.select:
            logger.info("reuse-existing specified without --select; nothing to do. Exiting.")
            return
        scores = df_target['boundary_score'].to_numpy()
        n = len(scores)
        idx_all = np.arange(n)
        if args.use_relative_margin:
            band_mask = (scores >= float(args.margin_low)) & (scores <= float(args.margin_high))
            candidates = idx_all[band_mask]
            if len(candidates) == 0:
                logger.warning("No candidates in margin band; falling back to global top by closeness to 1.0")
                candidates = idx_all
            order_local = np.argsort(np.abs(scores[candidates] - 1.0))
            idx_sel = candidates[order_local[:int(args.budget_total)]]
        else:
            order = np.argsort(scores)
            idx_sel = order[:int(args.budget_total)]

        role = df_target.get('role', pd.Series(['none'] * len(df_target)))
        role.iloc[:] = role.iloc[:].where(role != 'boundary', other='none')
        role.iloc[idx_sel] = 'boundary'
        df_target['role'] = role.values
        df_target.to_parquet(target_path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Reused existing boundary_score and selected {len(idx_sel)} rows (role='boundary') -> {target_path}")
        return

    # Load neighbor embeddings for full compute
    df_neighbor = _load_embeddings(neighbor_path, float32=args.float32)
    z_cols = [c for c in df_neighbor.columns if c.startswith('z_')]
    if not all(c in df_target.columns for c in z_cols):
        raise SystemExit("Mismatch embedding dimensions between neighbor and target")
    ZN = df_neighbor[z_cols].to_numpy()
    ZT = df_target[z_cols].to_numpy()
    logger.info(f"Neighbor Z: {ZN.shape}, Target Z: {ZT.shape}, k={args.k}, batch={args.batch_size}")

    # BoundarySelector
    selector = BoundarySelector(k=int(args.k), batch_size=int(args.batch_size), metric='euclidean')

    # Cross-class distances and collapse
    D_cross = selector.cross_distances(Z_target=ZT, Z_neighbor=ZN)
    d_cross = selector.collapse(D_cross, mode=args.score)

    # Score computation
    if args.use_relative_margin:
        # Same-class distances: cluster-local if cluster_id exists, else direct
        if 'cluster_id' in df_target.columns:
            d_same = selector.same_distances_cluster_local(Z_target=ZT, cluster_ids=df_target['cluster_id'].to_numpy())
        else:
            d_same = selector.same_distances_direct(Z_target=ZT)
        boundary_score = selector.relative_margin(d_cross, d_same)
        # Stats
        q = np.percentile(boundary_score, [10, 25, 50, 75, 90, 95, 99]).tolist()
        logger.info(f"Relative margin stats: p10={q[0]:.4f}, p25={q[1]:.4f}, p50={q[2]:.4f}, p75={q[3]:.4f}, p90={q[4]:.4f}, p95={q[5]:.4f}, p99={q[6]:.4f}")
    else:
        boundary_score = d_cross
        q = np.percentile(boundary_score, [50, 90, 95, 99]).tolist()
        logger.info(f"Boundary score stats: p50={q[0]:.6f}, p90={q[1]:.6f}, p95={q[2]:.6f}, p99={q[3]:.6f}")

    # Merge boundary_score into target embeddings (in-place)
    if 'RowId' in df_target.columns:
        df_target['boundary_score'] = boundary_score.astype(np.float32, copy=False)
    else:
        df_target['boundary_score'] = boundary_score

    # Optional selection using margin band + closest cross distance
    if args.select:
        idx_sel = selector.select(
            scores_margin=boundary_score if args.use_relative_margin else d_cross,
            d_cross_collapsed=d_cross,
            budget=int(args.budget_total),
            band_low=float(args.margin_low),
            band_high=float(args.margin_high),
        )
        role = df_target.get('role', pd.Series(['none'] * len(df_target)))
        # Reset previous boundary markings before applying new selection (align with --reuse-existing behavior)
        role.iloc[:] = role.iloc[:].where(role != 'boundary', other='none')
        role.iloc[idx_sel] = 'boundary'
        df_target['role'] = role.values
        logger.info(f"Marked role='boundary' for {len(idx_sel)} rows (selected PIN)")

    # Persist back to the same target embedding path
    df_target.to_parquet(target_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"Updated target embedding in-place -> {target_path} (added 'boundary_score'{', updated role' if args.select else ''})")


if __name__ == "__main__":
    main()