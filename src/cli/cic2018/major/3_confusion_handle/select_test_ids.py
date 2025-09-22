import os
import argparse
import numpy as np
import pandas as pd
import joblib

from utils.logging import setup_logging, get_logger
from sklearn.neighbors import NearestNeighbors
from configs import cic2018


logger = get_logger(__name__)


DATA_FOLDER = cic2018.DATA_FOLDER
EMBED_DIR_TEST = os.path.join(DATA_FOLDER, "embeddings", "test")
EMBED_DIR_TRAIN = os.path.join(DATA_FOLDER, "embeddings", "train")


def _load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    return pd.read_parquet(path)


def _load_centers(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise SystemExit(f"Centers file not found: {path}")
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        data = np.load(path)
        for key in ['centers', 'arr_0']:
            if key in data:
                return data[key]
        raise SystemExit(f"No centers array in npz: {list(data.keys())}")
    if path.endswith('.pkl'):
        arr = joblib.load(path)
        if isinstance(arr, np.ndarray):
            return arr
        raise SystemExit("Pickle does not contain a numpy array of centers")
    raise SystemExit("Unsupported centers file format; use .npy/.npz/.pkl")


def _assign_clusters(Z: np.ndarray, centers: np.ndarray, batch: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(centers.astype(np.float32, copy=False))
    labels = np.empty(len(Z), dtype=np.int32)
    dists = np.empty(len(Z), dtype=np.float32)
    try:
        from tqdm import tqdm
        it = tqdm(range(0, len(Z), batch), desc=f'assign clusters', total=(len(Z)+batch-1)//batch)
    except Exception:
        it = range(0, len(Z), batch)
    for s in it:
        e = min(s + batch, len(Z))
        d, idx = nn.kneighbors(Z[s:e], n_neighbors=1, return_distance=True)
        labels[s:e] = idx.ravel().astype(np.int32, copy=False)
        dists[s:e] = d.ravel().astype(np.float32, copy=False)
    return labels, dists


def _compute_boundary_scores(Z_target: np.ndarray, Z_neighbor: np.ndarray, batch: int = 200000) -> np.ndarray:
    """Compute boundary scores between target and neighbor embeddings."""
    nn_neighbor = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn_neighbor.fit(Z_neighbor)
    
    total = len(Z_target)
    d_cross = np.empty(total, dtype=np.float32)
    try:
        from tqdm import tqdm
        it_knn = tqdm(range(0, total, batch), desc=f'knn cross-class', total=(total+batch-1)//batch)
    except Exception:
        it_knn = range(0, total, batch)
    
    for s in it_knn:
        e = min(s + batch, total)
        d, _ = nn_neighbor.kneighbors(Z_target[s:e], n_neighbors=1, return_distance=True)
        d_cross[s:e] = d.ravel().astype(np.float32, copy=False)
    
    # Convert distance to boundary score (higher = closer to boundary)
    boundary_score = (1.0 / (1.0 + d_cross)).astype(np.float32, copy=False)
    return boundary_score


def main():
    parser = argparse.ArgumentParser(description="Balanced test selection: small/edge/boundary/core for both Benign and Infilteration")
    parser.add_argument('--label', type=str, default='Benign', choices=['Benign', 'Infilteration'])
    parser.add_argument('--budget-total', type=int, default=6000, help='Target total samples (default: 6000)')
    parser.add_argument('--target-embed', type=str, default=None,
                        help='Parquet embeddings for target test (auto-resolved if None)')
    parser.add_argument('--neighbor-embed', type=str, default=None,
                        help='Parquet embeddings for neighbor test (auto-resolved if None)')
    parser.add_argument('--centers-path', type=str, default=None,
                        help='KMeans centers from train (auto-resolved if None)')
    parser.add_argument('--benign-source', type=str, default='base', choices=['base', 'filtered'],
                        help='For Benign: choose centers variant corresponding to base/filtered embeddings')

    # Strategy params (adaptive based on label)
    parser.add_argument('--small-cluster-frac', type=float, default=None,
                        help='<= frac * total are small clusters (auto if None)')
    parser.add_argument('--edge-percentile', type=float, default=95.0, help='Keep points beyond this percentile per cluster (default: 95)')
    parser.add_argument('--boundary-frac', type=float, default=None,
                        help='Fraction of budget for boundary set (auto if None)')
    parser.add_argument('--large-rate', type=float, default=None,
                        help='Sampling rate for large clusters (auto if None)')
    parser.add_argument('--min-quota', type=int, default=8, help='Minimum per-cluster quota for large clusters (default: 8)')
    parser.add_argument('--knn-batch', type=int, default=200000)

    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--out-path', type=str, default=None,
                        help='Output parquet of selected RowIds (auto-resolved if None)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Debug: log hyperparameters
    logger.debug(f"Select test ids hyperparameters: label={args.label}, budget_total={args.budget_total}, small_cluster_frac={args.small_cluster_frac}, edge_percentile={args.edge_percentile}, boundary_frac={args.boundary_frac}, large_rate={args.large_rate}, min_quota={args.min_quota}, knn_batch={args.knn_batch}")

    # Auto-resolve paths
    target_safe = cic2018.get_label_name(args.label)
    neighbor_safe = cic2018.get_label_name('Infilteration' if args.label == 'Benign' else 'Benign')
    
    if args.target_embed is None:
        args.target_embed = os.path.join(EMBED_DIR_TEST, f"cic2018_{target_safe}_embedding.parquet")
    if args.neighbor_embed is None:
        args.neighbor_embed = os.path.join(EMBED_DIR_TEST, f"cic2018_{neighbor_safe}_embedding.parquet")
    if args.centers_path is None:
        benign_source = args.benign_source if args.label == 'Benign' else None
        args.centers_path = cic2018.kmeans_centers_path_train(args.label, benign_source=benign_source)
    if args.out_path is None:
        args.out_path = os.path.join(EMBED_DIR_TEST, f"cic2018_{target_safe}_test_selected_rowids.parquet")

    # Adaptive strategy parameters based on label
    if args.label == 'Benign':
        # Benign: heavily downsample (3M → 12k = 0.4%)
        small_cluster_frac = args.small_cluster_frac or 0.0003  # Very small clusters
        boundary_frac = args.boundary_frac or 0.12  # 12% for boundary
        large_rate = args.large_rate or 0.003  # 0.3% sampling rate
        logger.info(f"Benign strategy: small_frac={small_cluster_frac}, boundary_frac={boundary_frac}, large_rate={large_rate}")
    else:
        # Infilteration: moderate downsample (40k → 12k = 30%)
        small_cluster_frac = args.small_cluster_frac or 0.01  # Larger small clusters
        boundary_frac = args.boundary_frac or 0.10  # 10% for boundary
        large_rate = args.large_rate or 0.25  # 25% sampling rate
        logger.info(f"Infilteration strategy: small_frac={small_cluster_frac}, boundary_frac={boundary_frac}, large_rate={large_rate}")

    # Load embeddings
    target_df = _load_parquet(args.target_embed)
    neighbor_df = _load_parquet(args.neighbor_embed)
    logger.info(f"Loaded {args.label}_test={len(target_df)} {neighbor_safe}_test={len(neighbor_df)} | RowId present: target={('RowId' in target_df.columns)}")
    if 'RowId' not in target_df.columns:
        raise SystemExit(f"{args.label}_test embeddings have no RowId. Re-run umap_transform --subset test.")

    # Extract embeddings
    z_cols = [c for c in target_df.columns if c.startswith('z_')]
    Z_target = target_df[z_cols].to_numpy(dtype=np.float32 if args.float32 else None, copy=False)
    Z_neighbor = neighbor_df[[c for c in neighbor_df.columns if c.startswith('z_')]].to_numpy(dtype=np.float32 if args.float32 else None, copy=False)
    row_ids = target_df['RowId'].to_numpy()

    # Load and assign clusters
    logger.info(f"Loading centers: {args.centers_path}")
    centers = _load_centers(args.centers_path).astype(np.float32 if args.float32 else None, copy=False)
    logger.info(f"Centers shape: {centers.shape}")
    if centers.shape[1] != Z_target.shape[1]:
        raise SystemExit(f"Centers dim {centers.shape[1]} != embedding dim {Z_target.shape[1]}")

    labels, d2c = _assign_clusters(Z_target, centers, batch=args.knn_batch)
    target_df = target_df.copy()
    target_df['cluster_id'] = labels
    target_df['dist_center'] = d2c
    
    # Log distance stats
    try:
        p50 = float(np.percentile(d2c, 50)); p90 = float(np.percentile(d2c, 90)); p95 = float(np.percentile(d2c, 95))
        dmin = float(np.min(d2c)); dmax = float(np.max(d2c)); dmean = float(np.mean(d2c))
        logger.info(f"dist_center: min={dmin:.4f}, mean={dmean:.4f}, p50={p50:.4f}, p90={p90:.4f}, p95={p95:.4f}, max={dmax:.4f}")
    except Exception:
        pass

    # (1) Small clusters: keep 100%
    total = len(target_df)
    vc = target_df['cluster_id'].value_counts().to_dict()
    small_thr = max(1, int(np.ceil(small_cluster_frac * total)))
    small_clusters = {cid for cid, cnt in vc.items() if cnt <= small_thr}
    small_mask = target_df['cluster_id'].isin(small_clusters)
    logger.info(f"Small clusters threshold={small_thr} | num_small_clusters={len(small_clusters)} | points_in_small={int(small_mask.sum())}")

    # (2) Edge points: top percentile per cluster
    edge_mask = np.zeros(total, dtype=bool)
    edge_total = 0
    try:
        from tqdm import tqdm
        cluster_iter = tqdm(vc.items(), desc='edge per cluster', total=len(vc))
    except Exception:
        cluster_iter = vc.items()
    for cid, cnt in cluster_iter:
        idx = np.where(target_df['cluster_id'].to_numpy() == cid)[0]
        if len(idx) == 0:
            continue
        thr = np.percentile(target_df['dist_center'].to_numpy()[idx], args.edge_percentile)
        edge_idx = idx[target_df['dist_center'].to_numpy()[idx] >= thr]
        edge_mask[edge_idx] = True
        edge_total += int(len(edge_idx))
    logger.info(f"Edge points kept at p{args.edge_percentile}: total={edge_total}")

    # (3) Boundary points: closest to neighbor class
    boundary_score = _compute_boundary_scores(Z_target, Z_neighbor, batch=args.knn_batch)
    boundary_quota = max(1, int(args.budget_total * max(0.0, min(1.0, boundary_frac))))
    order = np.argsort(-boundary_score)
    boundary_idx = order[:boundary_quota]
    cutoff = float(boundary_score[boundary_idx[-1]]) if len(boundary_idx) > 0 else 0.0
    logger.info(f"Boundary quota={boundary_quota} | cutoff boundary_score={cutoff:.6f}")

    # (4) Compose initial selection
    selected = set(boundary_idx.tolist())
    selected.update(np.where(small_mask)[0].tolist())
    selected.update(np.where(edge_mask)[0].tolist())
    logger.info(f"Selected after boundary+small+edge: {len(selected)} (boundary={len(boundary_idx)}, small={int(small_mask.sum())}, edge={edge_total})")

    # (5) Trim if over budget
    if len(selected) > args.budget_total:
        excess = len(selected) - args.budget_total
        # Remove from edges first (excluding boundary and small)
        edge_only = [i for i in np.where(edge_mask)[0] if i not in boundary_idx and not small_mask[i]]
        if excess > 0 and edge_only:
            drop = min(excess, len(edge_only))
            selected.difference_update(edge_only[:drop])
            excess -= drop
            logger.info(f"Trimmed edge-only by {drop}, remaining_excess={excess}")
        # If still excess, remove from small_only (rare case)
        small_only = [i for i in np.where(small_mask)[0] if i not in boundary_idx and i not in edge_only]
        if excess > 0 and small_only:
            drop = min(excess, len(small_only))
            selected.difference_update(small_only[:drop])
            logger.info(f"Trimmed small-only by {drop}")
        logger.info(f"Selected after trim = {len(selected)} | target={args.budget_total}")

    # (6) Fill remaining budget with stratified sampling from large clusters
    if len(selected) < args.budget_total:
        need = args.budget_total - len(selected)
        rng = np.random.RandomState(42)
        remaining_mask = np.ones(total, dtype=bool)
        remaining_mask[list(selected)] = False
        remaining_idx = np.where(remaining_mask)[0]
        
        # Stratified by cluster
        per_cluster = {}
        try:
            from tqdm import tqdm
            strat_iter = tqdm(vc.items(), desc='stratify large clusters', total=len(vc))
        except Exception:
            strat_iter = vc.items()
        for cid, cnt in strat_iter:
            idx = [i for i in remaining_idx if target_df['cluster_id'].iat[i] == cid]
            if not idx:
                continue
            quota = max(args.min_quota, int(np.ceil(large_rate * len(idx))))
            quota = min(quota, len(idx))
            choose = rng.choice(idx, size=quota, replace=False)
            per_cluster[cid] = choose
        
        # Flatten and take up to need
        pool = np.concatenate(list(per_cluster.values())) if per_cluster else np.array([], dtype=int)
        if len(pool) > need:
            pool = pool[:need]
        selected.update(pool.tolist())
        logger.info(f"Added from large clusters: {min(len(pool), need)} | final_selected={len(selected)}")

    # In-place update: write computed columns back to original test parquet
    try:
        roles_full = np.array(['core'] * total, dtype=object)
        roles_full[boundary_idx] = 'boundary'
        roles_full[small_mask.to_numpy()] = 'small'
        roles_full[edge_mask] = 'edge'
        target_df['boundary_score'] = boundary_score.astype(np.float32, copy=False)
        target_df['role'] = roles_full
        # Persist back
        target_df.to_parquet(args.target_embed, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Wrote in-place to test parquet -> {args.target_embed} (added 'cluster_id', 'dist_center', 'boundary_score', 'role')")
    except Exception as e:
        logger.warning(f"Failed to write in-place updates to test parquet: {e}")

    # Build output
    selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
    
    # Final summary
    try:
        b_cnt = int(np.isin(selected_idx, boundary_idx).sum())
        s_cnt = int(small_mask[selected_idx].sum())
        e_cnt = int(edge_mask[selected_idx].sum())
        core_cnt = int(len(selected_idx) - b_cnt - s_cnt - e_cnt)
        logger.info(f"Final selected = {len(selected_idx)} / {args.budget_total} | boundary={b_cnt}, small={s_cnt}, edge={e_cnt}, core={core_cnt}")
    except Exception:
        logger.info(f"Final selected = {len(selected_idx)} / {args.budget_total}")
    
    # Assign roles
    roles = np.array(['core'] * len(selected_idx), dtype=object)
    roles[np.isin(selected_idx, boundary_idx)] = 'boundary'
    roles[small_mask[selected_idx]] = 'small'
    roles[edge_mask[selected_idx]] = 'edge'

    out_df = pd.DataFrame({
        'RowId': row_ids[selected_idx],
        'cluster_id': labels[selected_idx],
        'dist_center': d2c[selected_idx],
        'boundary_score': boundary_score[selected_idx],
        'role': roles,
    })
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out_df.to_parquet(args.out_path, engine='pyarrow', compression='snappy', index=False)
    
    # Role breakdown
    try:
        rvc = out_df['role'].value_counts().to_dict()
        logger.info(f"Saved selected RowIds -> {args.out_path} (rows={len(out_df)}) | role_breakdown={rvc}")
    except Exception:
        logger.info(f"Saved selected RowIds -> {args.out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
