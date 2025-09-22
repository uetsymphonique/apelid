import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector
from typing import Optional


logger = get_logger(__name__)


MAJOR_LABELS = cic2018.MAJORITY_LABELS


def _pca_parts_dir_candidates(subset: str, label_safe: str) -> list[str]:
    base = cic2018.PCA_CACHE_FOLDER
    return [
        os.path.join(base, f'cache_pca_{subset}', label_safe),
        os.path.join(base, subset, label_safe),
    ]


def _list_pca_part_files(subset: str, label_safe: str) -> list[str]:
    for cand in _pca_parts_dir_candidates(subset, label_safe):
        if os.path.isdir(cand):
            files = [
                os.path.join(cand, f)
                for f in sorted(os.listdir(cand))
                if f.endswith('.parquet') and f.startswith(f'pca_{subset}_{label_safe}_part_')
            ]
            if files:
                return files
    return []


def _default_n_clusters(n_rows: int, budget: int) -> int:
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


def _select_edges_per_cluster(dist_center: np.ndarray, labels: np.ndarray, percentile: float) -> dict[int, np.ndarray]:
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
    if need <= 0 or not edge_dict:
        return np.array([], dtype=np.int64)
    total_edges = int(sum(len(v) for v in edge_dict.values()))
    if total_edges == 0:
        return np.array([], dtype=np.int64)

    chosen: list[int] = []
    # proportional pass
    for c, v in edge_dict.items():
        if len(v) == 0:
            continue
        quota = int(round(need * (len(v) / total_edges)))
        take = min(quota, len(v))
        chosen.extend(v[:take].tolist())

    # round-robin remainder
    remaining = need - len(chosen)
    if remaining > 0:
        clusters = list(edge_dict.keys())
        pos = {c: 0 for c in clusters}
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


def _load_pca_df(subset: str, label: str, float32: bool) -> pd.DataFrame:
    label_safe = cic2018.get_label_name(label)
    files = _list_pca_part_files(subset, label_safe)
    if not files:
        raise SystemExit(f"No PCA parts found for {label} in subset {subset}. Expected under {cic2018.PCA_CACHE_FOLDER}/cache_pca_{subset}/{label_safe} or {cic2018.PCA_CACHE_FOLDER}/{subset}/{label_safe}")
    parts: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp)
        if 'RowId' not in df.columns:
            raise SystemExit(f"PCA part missing RowId: {fp}")
        if float32:
            pcols = [c for c in df.columns if c.startswith('pca_')]
            df[pcols] = df[pcols].astype(np.float32)
        parts.append(df)
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def _run_one(subset: str, label: str, budget: int, n_clusters: int | None, edge_percentile: float,
             batch_size: int, float32: bool, out_path: str | None) -> None:
    df = _load_pca_df(subset, label, float32=float32)
    label_safe = cic2018.get_label_name(label)
    pcols = [c for c in df.columns if c.startswith('pca_')]
    if not pcols:
        raise SystemExit("PCA columns not found (expected pca_*)")
    X = df[pcols].to_numpy(copy=False)
    n = len(df)
    logger.info(f"[{label}] PCA rows={n}, dims={len(pcols)} | budget={budget}")

    K = int(n_clusters) if n_clusters is not None else _default_n_clusters(n_rows=n, budget=int(budget))
    if K <= 0:
        raise SystemExit("n_clusters must be > 0")
    K = min(K, n)
    logger.info(f"[{label}] Using n_clusters={K}")

    ksel = KMeansRepresentativeSelector(n_clusters=K, batch_size=int(batch_size), random_state=42)
    labels_arr = ksel.fit_predict(X)
    centers = ksel.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")

    dist_center = _compute_dist_to_center(X, labels_arr, centers)

    # Core reps
    core_reps: list[int] = []
    for c in range(centers.shape[0]):
        idx = np.where(labels_arr == c)[0]
        if idx.size == 0:
            continue
        diff = X[idx] - centers[c]
        d2 = np.einsum('ij,ij->i', diff, diff)
        core_reps.append(int(idx[int(np.argmin(d2))]))
    core_reps = np.array(core_reps, dtype=np.int64)

    # Edge candidates
    edge_candidates = _select_edges_per_cluster(dist_center, labels_arr, percentile=float(edge_percentile))

    selected = set(core_reps.tolist())
    remaining = int(budget) - len(selected)

    chosen_edges = np.array([], dtype=np.int64)
    if remaining > 0:
        all_edge = np.concatenate(list(edge_candidates.values())) if edge_candidates else np.array([], dtype=np.int64)
        take_edge = min(remaining, len(all_edge))
        chosen_edges = _distribute_edges(edge_candidates, need=take_edge)
        if len(chosen_edges) > remaining:
            chosen_edges = chosen_edges[:remaining]
        selected.update(chosen_edges.tolist())
        remaining = int(budget) - len(selected)

    if remaining > 0:
        not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
        d_order = np.argsort(dist_center[not_sel])
        topup = not_sel[d_order[:remaining]]
        selected.update(topup.tolist())

    if len(selected) > int(budget):
        excess = len(selected) - int(budget)
        core_set = set(core_reps.tolist())
        edge_set = set(chosen_edges.tolist()) if chosen_edges.size > 0 else set()
        # Trim edge-only first (drop closer ones)
        edge_only = np.array([i for i in selected if (i in edge_set and i not in core_set)], dtype=np.int64)
        if edge_only.size > 0 and excess > 0:
            order = np.argsort(dist_center[edge_only])
            drop_k = min(excess, edge_only.size)
            to_drop = edge_only[order[:drop_k]].tolist()
            selected.difference_update(to_drop)
            excess -= drop_k
        # Then top-up
        if excess > 0:
            topup_only = np.array([i for i in selected if (i not in core_set and i not in edge_set)], dtype=np.int64)
            if topup_only.size > 0:
                order = np.argsort(dist_center[topup_only])
                drop_k = min(excess, topup_only.size)
                to_drop = topup_only[order[:drop_k]].tolist()
                selected.difference_update(to_drop)
                excess -= drop_k
        # Finally, core (rare)
        if excess > 0:
            core_only = np.array([i for i in selected if i in core_set], dtype=np.int64)
            if core_only.size > 0:
                order = np.argsort(dist_center[core_only])
                drop_k = min(excess, core_only.size)
                to_drop = core_only[order[:drop_k]].tolist()
                selected.difference_update(to_drop)
                logger.warning(f"[{label}] Trimmed {drop_k} core reps to meet budget")

    selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
    logger.info(f"[{label}] Final selected: {len(selected_idx)} / {budget} | clusters={K}")

    # Build output snapshot
    out_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    out_df['cluster_id'] = labels_arr[selected_idx].astype(np.int32, copy=False)
    out_df['dist_center'] = dist_center[selected_idx].astype(np.float32, copy=False)
    out_df['role'] = 'core'
    # Mark role edge where selected index is in edge candidates
    is_edge_selected = np.zeros(n, dtype=bool)
    for c, idx in edge_candidates.items():
        is_edge_selected[idx] = True
    mask_sel = np.zeros(n, dtype=bool); mask_sel[selected_idx] = True
    mask_edge = np.logical_and(mask_sel, is_edge_selected)
    out_df.loc[np.where(mask_edge[selected_idx])[0], 'role'] = 'edge'

    # Output path
    if out_path is None:
        if subset == 'train':
            out_p = os.path.join(cic2018.EMBEDDINGS_FOLDER, 'train', f"pca_cic2018_{label_safe}_compressed_coreset.parquet")
        else:
            out_p = os.path.join(cic2018.EMBEDDINGS_FOLDER, 'test', f"pca_cic2018_{label_safe}_test_selected_rowids.parquet")
    else:
        out_p = out_path
    os.makedirs(os.path.dirname(out_p), exist_ok=True)

    if subset == 'train':
        out_df.to_parquet(out_p, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"[{label}] Saved PCA compressed train snapshot -> {out_p} (rows={len(out_df)})")
    else:
        # For test, mimic 4_others: save RowIds only
        pd.DataFrame({'RowId': out_df['RowId'].astype(np.int64)}).to_parquet(
            out_p, engine='pyarrow', compression='snappy', index=False
        )
        logger.info(f"[{label}] Saved PCA selected test RowIds -> {out_p} (rows={len(out_df)})")


def main():
    parser = argparse.ArgumentParser(description="PCA-space KMeans compression for major classes (no UMAP, no confusion handling)")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, required=True, choices=MAJOR_LABELS + ['All'])
    parser.add_argument('--budget', type=int, required=True, help='Target compressed size per label')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (default: auto ~ budget)')
    parser.add_argument('--edge-percentile', type=float, default=95.0, help='Edge percentile per cluster (95..99)')
    parser.add_argument('--batch-size', type=int, default=10000, help='MiniBatchKMeans batch size')
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--out-path', type=str, default=None, help='Override output path')
    parser.add_argument('--n-jobs-labels', type=int, default=1, help='Parallel jobs across labels when --label All')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.label == 'All':
        labels = MAJOR_LABELS
        n_jobs = int(getattr(args, 'n_jobs_labels', 1) or 1)
        if n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                Parallel(n_jobs=n_jobs, backend='loky')([
                    delayed(_run_one)(
                        subset=args.subset,
                        label=lb,
                        budget=int(args.budget),
                        n_clusters=args.n_clusters,
                        edge_percentile=float(args.edge_percentile),
                        batch_size=int(args.batch_size),
                        float32=bool(args.float32),
                        out_path=None,
                    ) for lb in labels
                ])
                return
            except Exception as e:
                logger.warning(f"Parallel per-label failed ({e}); falling back to sequential")
        for lb in labels:
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


