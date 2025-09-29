import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from utils.logging import get_logger
import time



logger = get_logger(__name__)


class KMeansRepresentativeSelector:
    """
    Utility to fit MiniBatchKMeans and select representative points.

    Methods:
    - fit(X): fit MiniBatchKMeans on data X and store centers
    - predict(X): predict cluster labels for X
    - fit_predict(X): convenience to fit then return labels
    - select_representatives(X, labels=None, centers=None): per-cluster closest-to-centroid indices
    - assign_clusters_from_centers(X, centers): labels using provided centers
    - ensure_topup(indices, population_size, need, random_state=42): top-up indices to target size
    """

    def __init__(self, n_clusters: int, batch_size: int = 10000, random_state: int = 42, algorithm: str = 'minibatch'):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self._kmeans: MiniBatchKMeans | None = None
        self._centers: np.ndarray | None = None
        self.algorithm = 'full' if str(algorithm).lower() == 'full' else 'minibatch'

    @property
    def centers_(self) -> np.ndarray | None:
        return self._centers

    def fit(self, X: np.ndarray) -> None:
        time_start = time.time()
        if self.algorithm == 'full':
            logger.debug(f"[KMeansReps] Fitting KMeans(n_clusters={self.n_clusters})")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans.fit(X)
        else:
            logger.debug(f"[KMeansReps] Fitting MiniBatchKMeans(n_clusters={self.n_clusters}, batch_size={self.batch_size})")
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=self.random_state)
            kmeans.fit(X)
        self._kmeans = kmeans
        self._centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
        logger.info(f"[KMeansReps] Fit done. centers shape={self._centers.shape} in {time.time() - time_start:.2f} seconds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("KMeans not fitted. Call fit() first or use fit_predict().")
        return self._kmeans.predict(X).astype(np.int32, copy=False)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    @staticmethod
    def select_representatives(X: np.ndarray, labels: np.ndarray | None = None, centers: np.ndarray | None = None) -> np.ndarray:
        """
        Return indices of representative point (closest to center) per cluster.
        If labels/centers are None, raise error.
        """
        if labels is None or centers is None:
            raise ValueError("labels and centers are required to select representatives")
        reps: list[int] = []
        K = centers.shape[0]
        for c in range(K):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            diff = X[idx] - centers[c]
            d2 = np.einsum('ij,ij->i', diff, diff)
            reps.append(int(idx[int(np.argmin(d2))]))
        return np.array(reps, dtype=np.int64)

    @staticmethod
    def assign_clusters_from_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign cluster id by nearest center (L2)."""
        # Compute squared distances to all centers and choose argmin
        # (batching can be added if needed)
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1).astype(np.int32, copy=False)

    @staticmethod
    def ensure_topup(indices: np.ndarray, population_size: int, need: int, random_state: int = 42) -> np.ndarray:
        """
        Ensure the list of indices reaches target size `need` by random top-up
        without replacement from remaining population.
        """
        if need <= len(indices):
            return indices[:need]
        rng = np.random.RandomState(random_state)
        chosen = set(indices.tolist())
        candidates = [i for i in range(population_size) if i not in chosen]
        top_up = need - len(indices)
        if top_up > 0 and len(candidates) > 0:
            add_count = min(top_up, len(candidates))
            add_idx = rng.choice(candidates, size=add_count, replace=False)
            return np.concatenate([indices, add_idx.astype(np.int64)])
        return indices
