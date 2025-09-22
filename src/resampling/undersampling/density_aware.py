import numpy as np
import pandas as pd
from typing import Tuple, Dict

from utils.logging import setup_logging, get_logger

from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

logger = get_logger(__name__)


class DensityAwareFilter:
    def __init__(
        self,
        *,
        kmeans_k: int = 1500,
        kmeans_batch: int = 10000,
        small_cluster_frac: float = 0.001,
        edge_percentile: float = 90.0,
        large_keep_rate: float = 0.03,
        min_keep_per_large: int = 50,
        random_state: int = 42,
    ) -> None:
        self.kmeans_k = int(kmeans_k)
        self.kmeans_batch = int(kmeans_batch)
        self.small_cluster_frac = float(small_cluster_frac)
        self.edge_percentile = float(edge_percentile)
        self.large_keep_rate = float(large_keep_rate)
        self.min_keep_per_large = int(min_keep_per_large)
        self.random_state = int(random_state)

    def fit_microclusters(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        labels: np.ndarray
        centers: Dict[int, np.ndarray]

        logger.info(f"Using MiniBatchKMeans for micro-clustering")
        kmeans = MiniBatchKMeans(
            n_clusters=self.kmeans_k,
            batch_size=self.kmeans_batch,
            random_state=self.random_state,
        )
        logger.info(f"Fitting MiniBatchKMeans")
        labels = kmeans.fit_predict(Z)
        centers = {int(i): ctr for i, ctr in enumerate(kmeans.cluster_centers_)}
        return labels.astype(int), centers

    def filter_df(self, emb_df: pd.DataFrame, z_prefix: str = 'z_') -> pd.DataFrame:
        """Filter an embeddings DataFrame using density-aware rules.

        - Extracts embedding columns by prefix (default 'z_')
        - Fits micro-clusters
        - Computes distances to centroids
        - Applies density-aware keep mask
        """
        z_cols = [c for c in emb_df.columns if c.startswith(z_prefix)]
        if not z_cols:
            raise ValueError("Embedding columns not found (expected z_1..z_k)")

        Z = emb_df[z_cols].to_numpy(dtype=np.float32, copy=False)
        labels, centers = self.fit_microclusters(Z)
        logger.info(f"Micro-clustered {len(Z)} embeddings")
        # Attach cluster_id to dataframe for downstream QC/coverage checks
        emb_df = emb_df.copy()
        emb_df['cluster_id'] = labels
        try:
            # Brief coverage snapshot (top 10 clusters by size)
            vc = emb_df['cluster_id'].value_counts()
            top10 = vc.head(10).to_dict()
            logger.debug(f"Cluster distribution (top10): {top10}")
        except Exception:
            pass
        dist = self._distance_to_centroid(Z, labels, centers) if centers else np.zeros(len(Z), dtype=np.float32)
        logger.info(f"Calculated distances to centroid")
        keep = self._density_aware_keep_mask(
            labels=labels,
            distances=dist,
            total_rows=len(Z),
            outlier_label=-1,
        )
        kept = int(keep.sum())
        total = int(len(Z))
        ratio = (kept / max(1, total))
        logger.info(f"Calculated density-aware keep mask | Kept {kept} / {total} ({ratio:.2%})")
        return emb_df[keep].reset_index(drop=True)

    @staticmethod
    def _compute_cluster_stats(labels: np.ndarray) -> Dict[int, int]:
        stats: Dict[int, int] = {}
        unique, counts = np.unique(labels, return_counts=True)
        for lb, cnt in zip(unique, counts):
            stats[int(lb)] = int(cnt)
        return stats

    @staticmethod
    def _distance_to_centroid(Z: np.ndarray, labels: np.ndarray, centers: Dict[int, np.ndarray]) -> np.ndarray:
        d = np.zeros(len(Z), dtype=np.float32)
        for i in tqdm(range(len(Z)), desc='dist->centroid', total=len(Z)):
            lb = int(labels[i])
            if lb in centers:
                ctr = centers[lb]
                d[i] = np.linalg.norm(Z[i] - ctr)
            else:
                d[i] = 0.0
        return d

    def _density_aware_keep_mask(
        self,
        *,
        labels: np.ndarray,
        distances: np.ndarray,
        total_rows: int,
        outlier_label: int = -1,
    ) -> np.ndarray:
        """Compute boolean keep mask per point using density-aware rules."""
        stats = self._compute_cluster_stats(labels)
        small_threshold = max(1, int(np.ceil(self.small_cluster_frac * total_rows)))
        rng = np.random.RandomState(self.random_state)

        keep = np.zeros(len(labels), dtype=bool)

        # 1) Keep all outliers/noise
        if outlier_label in stats:
            noise_mask = (labels == outlier_label)
            noise_cnt = int(noise_mask.sum())
            keep |= noise_mask
            logger.debug(f"Noise/outliers kept: {noise_cnt}")

        # 2) Keep small clusters entirely
        items = list(stats.items())
        small_total = 0
        for lb, cnt in tqdm(items, desc='keep small clusters', total=len(items)):
            if lb == outlier_label:
                continue
            if cnt <= small_threshold:
                idx = (labels == lb)
                small_total += int(idx.sum())
                keep |= idx
        logger.debug(f"Small clusters threshold={small_threshold} | points kept: {small_total}")

        # 3) Keep edge points beyond percentile within cluster
        items = list(stats.items())
        edge_added = 0
        for lb, cnt in tqdm(items, desc='keep edge points', total=len(items)):
            if lb == outlier_label:
                continue
            idx = np.where(labels == lb)[0]
            if len(idx) == 0:
                continue
            thr = np.percentile(distances[idx], self.edge_percentile)
            edge_idx = idx[distances[idx] >= thr]
            newly = np.setdiff1d(edge_idx, np.where(keep)[0], assume_unique=False)
            edge_added += int(len(newly))
            keep[edge_idx] = True
        logger.debug(f"Edge points kept at {self.edge_percentile}th percentile | newly added: {edge_added}")

        # 4) Downsample in large clusters (excluding already kept)
        items = list(stats.items())
        down_added = 0
        for lb, cnt in tqdm(items, desc='downsample large clusters', total=len(items)):
            if lb == outlier_label:
                continue
            if cnt > small_threshold:
                idx = np.where(labels == lb)[0]
                candidates = idx[~keep[idx]]
                if len(candidates) == 0:
                    continue
                target = max(self.min_keep_per_large, int(np.ceil(self.large_keep_rate * len(idx))))
                target = min(target, len(candidates))
                if target > 0:
                    chosen = rng.choice(candidates, size=target, replace=False)
                    keep[chosen] = True
                    down_added += int(len(chosen))
        logger.debug(f"Downsampled large clusters | newly added: {down_added}")

        return keep
    # No free-function API: use class-based API only.


