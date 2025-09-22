import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.logging import get_logger


logger = get_logger(__name__)


class BoundarySelector:
    """
    Dataset-agnostic boundary scoring and selection.

    Core ideas:
    - Cross-class proximity via kNN to neighbor set (Euclidean by default)
    - Same-class proximity via either:
        * direct kNN on target (small/medium sets), or
        * cluster-based local kNN using provided cluster_ids (for very large sets)
    - Relative margin m = d_cross / (d_same + eps)

    API focuses on numpy arrays to be reusable across datasets/pipelines.
    """

    def __init__(self,
                 k: int = 10,
                 batch_size: int = 100000,
                 metric: str = 'euclidean',
                 random_state: int = 42):
        self.k = int(k)
        self.batch_size = int(batch_size)
        self.metric = metric
        self.random_state = int(random_state)

    # ---------- Cross-class distances ----------
    def cross_distances(self, Z_target: np.ndarray, Z_neighbor: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        nn.fit(Z_neighbor)
        n = len(Z_target)
        D = np.empty((n, self.k), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            d, _ = nn.kneighbors(Z_target[start:end], n_neighbors=self.k, return_distance=True)
            D[start:end] = d.astype(np.float32, copy=False)
        return D

    def collapse(self, D: np.ndarray, mode: str = 'min') -> np.ndarray:
        if mode == 'min':
            return D[:, 0]
        if mode == 'mean':
            return D.mean(axis=1)
        if mode == 'p95':
            return np.percentile(D, 95.0, axis=1)
        raise ValueError("mode must be one of {'min','mean','p95'}")

    # ---------- Same-class distances ----------
    def same_distances_direct(self, Z_target: np.ndarray) -> np.ndarray:
        n = len(Z_target)
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n), metric=self.metric)
        nn.fit(Z_target)
        d_same = np.empty(n, dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            d, _ = nn.kneighbors(Z_target[start:end], n_neighbors=min(self.k + 1, end - start), return_distance=True)
            d_same[start:end] = d[:, 1] if d.shape[1] > 1 else d[:, 0]
        return d_same

    def same_distances_cluster_local(self, Z_target: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Approximate same-class distance using local kNN within each cluster.
        cluster_ids: shape (n,), integers per row. Assumes each id denotes a local neighborhood.
        """
        d_same = np.empty(len(Z_target), dtype=np.float32)
        unique = np.unique(cluster_ids)
        for cid in unique:
            mask = (cluster_ids == cid)
            Zc = Z_target[mask]
            m = len(Zc)
            if m == 0:
                continue
            if m == 1:
                d_same[mask] = 0.0
                continue
            nn = NearestNeighbors(n_neighbors=min(self.k + 1, m), metric=self.metric)
            nn.fit(Zc)
            d, _ = nn.kneighbors(Zc, n_neighbors=min(self.k + 1, m), return_distance=True)
            d_same[mask] = d[:, 1] if d.shape[1] > 1 else d[:, 0]
        return d_same

    # ---------- Scoring and selection ----------
    @staticmethod
    def relative_margin(d_cross: np.ndarray, d_same: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return d_cross / (d_same + float(eps))

    def select(self,
               scores_margin: np.ndarray,
               d_cross_collapsed: np.ndarray,
               budget: int,
               band_low: float = 0.5,
               band_high: float = 2.0) -> np.ndarray:
        """
        Select boundary samples: filter by margin band, then choose the smallest cross-distance inside the band.
        Returns indices into the provided arrays.
        """
        if budget <= 0:
            return np.array([], dtype=np.int64)
        mask = (scores_margin >= band_low) & (scores_margin <= band_high)
        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0:
            # Fallback: choose by d_cross globally
            order = np.argsort(d_cross_collapsed)
            return order[:min(budget, len(order))]
        # Rank candidates by d_cross within the band
        order_local = np.argsort(d_cross_collapsed[cand_idx])
        chosen_local = order_local[:min(budget, len(order_local))]
        return cand_idx[chosen_local].astype(np.int64)
