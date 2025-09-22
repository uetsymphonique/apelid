import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from ..helpers import load_embeddings
from resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector


logger = get_logger(__name__)


MAJOR_LABELS = [
    'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk',
    'Bot', 'SSH-Bruteforce', 'DoS attacks-GoldenEye'
]


def _default_n_clusters(n_rows: int, budget: int) -> int:
    """
    Heuristic for number of clusters:
    - Prefer n_clusters ~= budget
    - If dataset is much larger (>5x budget), use budget/2 to keep clusters stable,
      then we will fill the rest with edge points.
    """
    if budget <= 0:
        return 0
    if n_rows > 5 * budget:
        return max(1, budget // 2)
    return min(budget, n_rows)


def _compute_dist_to_center(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist = np.empty(len(X), dtype=np.float32)
    for c in range(centers.shape[0]):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        diff = X[idx] - centers[c]
        d2 = np.einsum('ij,ij->i', diff, diff)
        dist[idx] = np.sqrt(d2).astype(np.float32, copy=False)
    return dist


def _per_cluster_representative_core(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    reps = []
    K = centers.shape[0]
    for c in range(K):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        diff = X[idx] - centers[c]
        d2 = np.einsum('ij,ij->i', diff, diff)
        reps.append(idx[int(np.argmin(d2))])
    return np.array(reps, dtype=np.int64)


def _select_edges_per_cluster(dist_center: np.ndarray, labels: np.ndarray, percentile: float) -> dict[int, np.ndarray]:
    """Return dict cluster_id -> indices of edge candidates (>= percentile distance)."""
    edge_dict: dict[int, np.ndarray] = {}
    unique_clusters = np.unique(labels)
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        thr = np.percentile(dist_center[idx], percentile)
        edge_idx = idx[dist_center[idx] >= thr]
        edge_dict[int(c)] = edge_idx
    return edge_dict


def _distribute_edges(edge_dict: dict[int, np.ndarray], need: int) -> np.ndarray:
    """
    Proportional allocation of edge picks across clusters.
    - First pass: proportionate to available edge candidates
    - Second pass: round-robin 1-by-1 while need remains
    """
    if need <= 0 or not edge_dict:
        return np.array([], dtype=np.int64)
    total_edges = int(sum(len(v) for v in edge_dict.values()))
    if total_edges == 0:
        return np.array([], dtype=np.int64)

    chosen: list[int] = []
    # Sort edge candidates by distance descending within cluster for stability
    sorted_edges = {c: v[np.argsort(-v.size + np.arange(v.size))] if len(v) > 0 else v for c, v in edge_dict.items()}

    # Proportional first pass
    for c, v in edge_dict.items():
        if len(v) == 0:
            continue
        quota = int(round(need * (len(v) / total_edges)))
        take = min(quota, len(v))
        chosen.extend(v[:take].tolist())

    # If still short, round-robin remaining
    remaining = need - len(chosen)
    if remaining > 0:
        clusters = list(edge_dict.keys())
        pos = {c: len([i for i in chosen if i in edge_dict[c]]) for c in clusters}
        while remaining > 0:
            progressed = False
            for c in clusters:
                v = edge_dict[c]
                if pos[c] < len(v):
                    chosen.append(int(v[pos[c]]))
                    pos[c] += 1
                    remaining -= 1
                    progressed = True
                    if remaining <= 0:
                        break
            if not progressed:
                break

    return np.unique(np.array(chosen, dtype=np.int64))


def _run_one(subset: str, label: str, budget: int, n_clusters: int | None, edge_percentile: float,
             batch_size: int, float32: bool, out_path: str | None) -> None:
    # Resolve input path for this label
    path = cic2018.embedding_path(subset, label, filtered_benign=False)
    if not os.path.exists(path):
        raise SystemExit(f"Embedding parquet not found: {path}")

    df = load_embeddings(path, float32=float32)
    if 'RowId' not in df.columns:
        raise SystemExit("Embeddings must contain RowId (ensure umap_transform added it)")
    z_cols = [c for c in df.columns if c.startswith('z_')]
    if not z_cols:
        raise SystemExit("Embedding columns not found (expected z_*)")
    X = df[z_cols].to_numpy()
    n = len(df)
    logger.info(f"[{label}] rows={n}, dims={len(z_cols)} | budget={budget}")

    # Determine clusters
    K = int(n_clusters) if n_clusters is not None else _default_n_clusters(n_rows=n, budget=int(budget))
    if K <= 0:
        raise SystemExit("n_clusters must be > 0")
    K = min(K, n)
    logger.info(f"[{label}] Using n_clusters={K}")

    # Fit KMeans and assign labels
    ksel = KMeansRepresentativeSelector(n_clusters=K, batch_size=int(batch_size), random_state=42)
    labels_arr = ksel.fit_predict(X)
    centers = ksel.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")

    # Compute dist_center
    dist_center = _compute_dist_to_center(X, labels_arr, centers)

    # In-place add cluster_id and dist_center
    try:
        df_out = pd.read_parquet(path)
        df_out['cluster_id'] = labels_arr.astype(np.int32, copy=False)
        df_out['dist_center'] = dist_center.astype(np.float32, copy=False)
        df_out.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"[{label}] Wrote in-place: cluster_id, dist_center -> {path}")
    except Exception as e:
        logger.warning(f"[{label}] Failed to write in-place cluster columns: {e}")

    # Core representatives (1 per cluster)
    core_reps = _per_cluster_representative_core(X, labels_arr, centers)

    # Edge candidates per cluster
    edge_candidates = _select_edges_per_cluster(dist_center, labels_arr, percentile=float(edge_percentile))

    # Assemble selection: ensure coverage first
    selected = set(core_reps.tolist())
    remaining = int(budget) - len(selected)

    if remaining > 0:
        # Distribute edge picks proportionally
        all_edge = np.concatenate(list(edge_candidates.values())) if edge_candidates else np.array([], dtype=np.int64)
        take_edge = min(remaining, len(all_edge))
        chosen_edges = _distribute_edges(edge_candidates, need=take_edge)
        # Clamp to remaining to avoid overshooting
        if len(chosen_edges) > remaining:
            chosen_edges = chosen_edges[:remaining]
        selected.update(chosen_edges.tolist())
        remaining = int(budget) - len(selected)

    if remaining > 0:
        # Top-up with nearest-to-center not yet chosen
        not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
        d_order = np.argsort(dist_center[not_sel])  # nearest cores first
        topup = not_sel[d_order[:remaining]]
        selected.update(topup.tolist())

    # Safety trim: if still above budget, drop lowest-priority points (edge/topup) first
    if len(selected) > int(budget):
        excess = len(selected) - int(budget)
        core_set = set(core_reps.tolist())
        edge_set = set(chosen_edges.tolist()) if 'chosen_edges' in locals() else set()

        # 1) Trim from edge-only (keep more extreme edges by dist_center)
        edge_only = np.array([i for i in selected if (i in edge_set and i not in core_set)], dtype=np.int64)
        if edge_only.size > 0 and excess > 0:
            order = np.argsort(dist_center[edge_only])  # drop closer-to-center edges first
            drop_k = min(excess, edge_only.size)
            to_drop = edge_only[order[:drop_k]].tolist()
            selected.difference_update(to_drop)
            excess -= drop_k
            logger.info(f"[{label}] Trimmed {drop_k} edge picks to meet budget")

        # 2) Trim from top-up (non-core, non-edge), drop nearest-to-center first
        if excess > 0:
            topup_only = np.array([i for i in selected if (i not in core_set and i not in edge_set)], dtype=np.int64)
            if topup_only.size > 0:
                order = np.argsort(dist_center[topup_only])
                drop_k = min(excess, topup_only.size)
                to_drop = topup_only[order[:drop_k]].tolist()
                selected.difference_update(to_drop)
                excess -= drop_k
                logger.info(f"[{label}] Trimmed {drop_k} top-up picks to meet budget")

        # 3) As last resort, if still excess, trim from core (rare). Drop nearest-to-center first
        if excess > 0:
            core_only = np.array([i for i in selected if i in core_set], dtype=np.int64)
            if core_only.size > 0:
                order = np.argsort(dist_center[core_only])
                drop_k = min(excess, core_only.size)
                to_drop = core_only[order[:drop_k]].tolist()
                selected.difference_update(to_drop)
                excess -= drop_k
                logger.warning(f"[{label}] Trimmed {drop_k} core reps (fallback) to meet budget")

    selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
    logger.info(f"[{label}] Final selected: {len(selected_idx)} / {budget} | clusters={K}")

    # Build output
    out_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    out_df['role'] = 'core'
    # Mark role='edge' where selected and edge by threshold
    is_edge_selected = np.zeros(len(df), dtype=bool)
    for c, idx in edge_candidates.items():
        is_edge_selected[idx] = True
    mask_sel = np.zeros(n, dtype=bool); mask_sel[selected_idx] = True
    mask_edge = np.logical_and(mask_sel, is_edge_selected)
    out_df.loc[np.where(mask_edge[selected_idx])[0], 'role'] = 'edge'

    # Output path per label
    if out_path is None:
        label_safe = cic2018.get_label_name(label)
        if subset == 'train':
            base, ext = os.path.splitext(path)
            out_p = f"{base}_compressed_coreset{ext or '.parquet'}"
        else:
            test_dir = os.path.join(cic2018.DATA_FOLDER, 'embeddings', 'test')
            os.makedirs(test_dir, exist_ok=True)
            out_p = os.path.join(test_dir, f"cic2018_{label_safe}_test_selected_rowids.parquet")
    else:
        out_p = out_path
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    out_df.to_parquet(out_p, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"[{label}] Saved compressed set -> {out_p} (rows={len(out_df)})")

def main():
    parser = argparse.ArgumentParser(description="Simple compression for major classes using KMeans cores + edge per cluster")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, required=True,
        choices=MAJOR_LABELS + ['All'])
    parser.add_argument('--input-path', type=str, default=None,
                        help='Embedding parquet (default resolves from embeddings/<subset>/)')
    parser.add_argument('--budget', type=int, required=True, help='Target compressed size')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (default: auto ~ budget)')
    parser.add_argument('--edge-percentile', type=float, default=95.0, help='Percentile for edge per cluster (95..99)')
    parser.add_argument('--batch-size', type=int, default=10000, help='MiniBatchKMeans batch size')
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--out-path', type=str, default=None, help='Output parquet for compressed set (default: base + _compressed_major)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # All-mode: iterate labels, ignore input/out overrides
    if args.label == 'All':
        for lb in MAJOR_LABELS:
            try:
                _run_one(
                    subset=args.subset,
                    label=lb,
                    budget=int(args.budget),
                    n_clusters=args.n_clusters,
                    edge_percentile=float(args.edge_percentile),
                    batch_size=int(args.batch_size),
                    float32=bool(args.float32),
                    out_path=None,
                )
            except SystemExit as e:
                logger.warning(f"[{lb}] skipped: {e}")
        return

    # Single label mode
    _run_one(
        subset=args.subset,
        label=args.label,
        budget=int(args.budget),
        n_clusters=args.n_clusters,
        edge_percentile=float(args.edge_percentile),
        batch_size=int(args.batch_size),
        float32=bool(args.float32),
        out_path=args.out_path,
    )



if __name__ == "__main__":
    main()


