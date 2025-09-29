import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


MAJOR_LABELS = cic2018.MAJORITY_LABELS


def _encoded_path(subset: str, label: str) -> str:
    label_safe = cic2018.get_label_name(label)
    base_dir = os.path.join(cic2018.ENCODED_DATA_FOLDER, subset)
    return os.path.join(base_dir, f"cic2018_{label_safe}_encoded.csv")


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


def _load_encoded_df(subset: str, label: str) -> pd.DataFrame:
    path = _encoded_path(subset, label)
    if not os.path.exists(path):
        raise SystemExit(f"Encoded CSV not found for {label} in subset {subset}: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


# (subset='all' mode removed)


def _run_one(subset: str, label: str, budget: int, n_clusters: int | None, edge_percentile: float,
             batch_size: int, out_path: str | None, strategy: str = 'diverse', avg_pts_per_cluster: float = 3.0,
             core_fallback: str = 'edge_topup', kmeans_algo: str = 'minibatch') -> None:
    df = _load_encoded_df(subset, label)
    label_safe = cic2018.get_label_name(label)
    # Use the encoded numerical feature list from the preprocessor (StandardScaler space)
    pre = CIC2018Preprocessor()
    # Exclude technical columns from features
    tech_cols = {'__rowid__', '__subset__'}
    feat_cols = [c for c in pre.encoded_numerical_features if c in df.columns and c not in tech_cols]
    if not feat_cols:
        raise SystemExit("No encoded numerical feature columns found in encoded CSV")
    X = df[feat_cols].to_numpy(copy=False)
    n = len(df)
    logger.info(f"[{label}] strategy={strategy} | rows={n}, dims={len(feat_cols)} | budget={budget}")

    if n_clusters is not None:
        K = int(n_clusters)
    else:
        if strategy == 'core':
            K = int(budget)
        else:
            # diverse: K ~ budget / avg_pts_per_cluster
            K = int(max(1, min(n, budget // max(1.0, float(avg_pts_per_cluster)))))
    if K <= 0:
        raise SystemExit("n_clusters must be > 0")
    K = min(K, n)
    logger.info(f"[{label}] Using n_clusters={K}")

    ksel = KMeansRepresentativeSelector(n_clusters=K, batch_size=int(batch_size), random_state=42, algorithm=kmeans_algo)
    labels_arr = ksel.fit_predict(X)
    centers = ksel.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")

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

    if strategy == 'core':
        selected_idx = core_reps
        if len(selected_idx) < int(budget):
            if core_fallback == 'random':
                logger.info(f"[{label}] core-only selected {len(selected_idx)} < budget {budget}. Falling back to random fill …")
                selected = set(core_reps.tolist())
                remaining = int(budget) - len(selected)
                not_sel = np.array(sorted(list(set(range(n)) - selected)), dtype=np.int64)
                take = min(remaining, not_sel.size)
                if take > 0:
                    rng = np.random.RandomState(42)
                    extra = rng.choice(not_sel, size=take, replace=False)
                    selected.update(extra.tolist())
                selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
                logger.info(f"[{label}] Final selected (core+random): {len(selected_idx)} / {budget} | clusters={K}")
            else:
                logger.info(f"[{label}] core-only selected {len(selected_idx)} < budget {budget}. Falling back to edge + top-up …")
                # Compute distances and supplement by edge then top-up
                dist_center = _compute_dist_to_center(X, labels_arr, centers)
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
                    if not_sel.size > 0:
                        d_order = np.argsort(dist_center[not_sel])
                        topup = not_sel[d_order[:remaining]]
                        selected.update(topup.tolist())

                selected_idx = np.array(sorted(list(selected)), dtype=np.int64)
                logger.info(f"[{label}] Final selected (core+fallback): {len(selected_idx)} / {budget} | clusters={K}")
        else:
            logger.info(f"[{label}] Final selected (core-only): {len(selected_idx)} / {budget} | clusters={K}")
    else:
        # Diverse mode: compute distances and edge selection
        dist_center = _compute_dist_to_center(X, labels_arr, centers)

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
        logger.info(f"[{label}] Final selected (diverse): {len(selected_idx)} / {budget} | clusters={K}")

    # Build output: selected encoded rows only (preserve __rowid__ and __subset__ if present)
    out_df = df.iloc[selected_idx].copy().reset_index(drop=True)

    # Map back to clean_merged using __rowid__ and write clean_merged compressed for the same subset
    src_path = os.path.join(cic2018.CLEAN_MERGED_DATA_FOLDER, subset, f"cic2018_{label_safe}_{subset}_clean_merged.csv")
    if not os.path.exists(src_path):
        raise SystemExit(f"Source clean_merged not found: {src_path}")
    src_df = pd.read_csv(src_path, low_memory=False)
    if '__rowid__' not in out_df.columns:
        raise SystemExit("Missing __rowid__ in selection; re-run encode to embed row ids.")
    sel_ids = [int(i) for i in out_df['__rowid__'].astype(np.int64).tolist() if 0 <= int(i) < len(src_df)]
    cm_df = src_df.iloc[sel_ids].copy()
    for c in ['__rowid__', '__subset__']:
        if c in cm_df.columns:
            cm_df = cm_df.drop(columns=[c])
    out_p = out_path if out_path is not None else os.path.join(
        cic2018.CLEAN_MERGED_DATA_FOLDER, subset, f"cic2018_{label_safe}_{subset}_clean_merged_compressed.csv"
    )
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    cm_df.to_csv(out_p, index=False)
    logger.info(f"[{label}] Saved clean_merged compressed -> {out_p} (rows={len(cm_df)})")


def main():
    parser = argparse.ArgumentParser(description="KMeans compression on encoded CSV for major classes (no PCA/UMAP)")
    parser.add_argument('--subset', '-s', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'], help='Compress all labels or only provided labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=MAJOR_LABELS, help='Label names when --label-mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=MAJOR_LABELS, help='Labels to exclude after selection')
    parser.add_argument('--budget', '-b', type=int, required=True, help='Target compressed size per label')
    parser.add_argument('--n-clusters', '-k', type=int, default=None, help='Number of clusters (default: auto ~ budget)')
    parser.add_argument('--edge-percentile', '-e', type=float, default=95.0, help='Edge percentile per cluster (95..99)')
    parser.add_argument('--batch-size', type=int, default=10000, help='MiniBatchKMeans batch size')
    parser.add_argument('--out-path', '-o', type=str, default=None, help='Override output path')
    parser.add_argument('--n-jobs-labels', '-j', type=int, default=1, help='Parallel jobs across labels when --label All')
    parser.add_argument('--strategy', '-t', type=str, default='diverse', choices=['core', 'diverse'], help='Selection strategy: core-only or diverse (core+edge)')
    parser.add_argument('--avg-pts-per-cluster', '-p', type=float, default=3.0, help='Diverse mode: average selected points per cluster (K ≈ budget/avg)')
    parser.add_argument('--core-fallback', type=str, default='edge_topup', choices=['edge_topup', 'random'], help='Fallback method when strategy=core under-selects: edge_topup or random')
    parser.add_argument('--kmeans-algo', type=str, default='minibatch', choices=['minibatch', 'full'], help='Clustering algorithm: MiniBatchKMeans (default) or KMeans (full)')
    # removed --split-frac (no subset=all)
    parser.add_argument('--log-level', '-L', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --label-mode label")
        labels = args.labels
    else:
        labels = MAJOR_LABELS

    # Apply excludes if provided
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        labels = [lb for lb in labels if lb not in excludes]
        logger.info(f"[+] Excluding labels: {sorted(excludes)}")

    if not labels:
        raise SystemExit("No labels to process after applying excludes.")

    n_jobs = int(getattr(args, 'n_jobs_labels', 1) or 1)
    if n_jobs != 1 and len(labels) > 1:
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
                    out_path=None,
                    strategy=args.strategy,
                    avg_pts_per_cluster=float(args.avg_pts_per_cluster),
                    core_fallback=args.core_fallback,
                    kmeans_algo=args.kmeans_also if hasattr(args, 'kmeans_also') else args.kmeans_algo) for lb in labels
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
                out_path=args.out_path if (len(labels) == 1) else None,
                strategy=args.strategy,
                avg_pts_per_cluster=float(args.avg_pts_per_cluster),
                core_fallback=args.core_fallback,
                kmeans_algo=args.kmeans_algo,
            )
        except SystemExit as e:
            logger.warning(f"[{lb}] skipped: {e}")


if __name__ == "__main__":
    main()