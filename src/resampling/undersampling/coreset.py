import numpy as np
from typing import Tuple
from utils.logging import get_logger
from .kmeans_reps import KMeansRepresentativeSelector


logger = get_logger(__name__)


class CoresetSelector:
    """
    Balanced coreset selection:
    - Start with PIN (pre-selected boundary indices)
    - Split non-PIN by margin threshold into core candidates and overlap candidates
    - Select core representatives using MiniBatchKMeans (closest-to-centroid),
      supporting multiple representatives per cluster when budget_core > K
    - Select overlap by random sampling (small quota for diversity)
    - Top-up core if clusters yield fewer reps than needed

    Inputs operate on numpy arrays to keep dataset-agnostic.
    """

    def __init__(self,
                 kmeans_batch: int = 10000,
                 random_state: int = 42):
        self.kmeans_batch = int(kmeans_batch)
        self.random_state = int(random_state)

    def select(self,
               X: np.ndarray,
               pin_idx: np.ndarray,
               margin_values: np.ndarray | None,
               n_clusters: int | None,
               budget_total: int,
               overlap_ratio: float = 0.05,
               min_margin: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (selected_idx, pin_idx, core_idx, overlap_idx).
        - X: (N, D) embeddings
        - pin_idx: indices of PIN (can be empty)
        - margin_values: per-row scores (e.g., relative margin). If None => no margin split
        - n_clusters: number of clusters to use for KMeans
        - budget_total: final target size
        - overlap_ratio: fraction of remaining budget for overlap
        - min_margin: threshold to split core vs overlap when margin_values provided
        """
        n = len(X)
        if budget_total <= 0:
            return np.array([], dtype=np.int64), pin_idx, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        pin_idx = np.unique(pin_idx.astype(np.int64)) if pin_idx.size > 0 else np.array([], dtype=np.int64)
        remaining = budget_total - len(pin_idx)
        if remaining <= 0:
            logger.info("PIN already fills the budget")
            return pin_idx.copy(), pin_idx, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        budget_overlap = int(max(0, min(1.0, overlap_ratio)) * remaining)
        budget_core = remaining - budget_overlap

        # Build non-PIN pool
        nonpin_mask = np.ones(n, dtype=bool)
        if pin_idx.size > 0:
            nonpin_mask[pin_idx] = False
        nonpin_idx = np.where(nonpin_mask)[0]

        # Margin-based split
        if margin_values is not None:
            mv = np.asarray(margin_values, dtype=np.float32)
            mv = mv[nonpin_idx]
            core_mask = mv >= float(min_margin)
            overlap_mask = ~core_mask
            core_candidates = nonpin_idx[core_mask]
            overlap_candidates = nonpin_idx[overlap_mask]
        else:
            core_candidates = nonpin_idx
            overlap_candidates = np.array([], dtype=np.int64)

        # Core via KMeans reps (multiple reps per cluster if needed)
        core_idx = np.array([], dtype=np.int64)
        if budget_core > 0 and core_candidates.size > 0:
            K = int(n_clusters) if n_clusters is not None else int(min(budget_core, max(1, core_candidates.size)))
            selector = KMeansRepresentativeSelector(n_clusters=K, batch_size=self.kmeans_batch, random_state=self.random_state)
            X_core = X[core_candidates]
            labels = selector.fit_predict(X_core)
            centers = selector.centers_
            if centers is None:
                centers = np.zeros((K, X_core.shape[1]), dtype=np.float32)

            # Per-cluster indices
            cluster_to_indices: list[np.ndarray] = []
            for c in range(K):
                idx = np.where(labels == c)[0]
                cluster_to_indices.append(idx)

            # Base quota per cluster and remainder
            base_q = max(1, budget_core // K)
            # If base_q==0 (shouldn't due to max), keep safety
            chosen_local = []
            for c in range(K):
                idx = cluster_to_indices[c]
                if idx.size == 0:
                    continue
                # distances to center
                diff = X_core[idx] - centers[c]
                d2 = np.einsum('ij,ij->i', diff, diff)
                order = np.argsort(d2)
                take = min(base_q, idx.size)
                chosen_local.append(idx[order[:take]])
            chosen = np.concatenate(chosen_local) if chosen_local else np.array([], dtype=np.int64)

            # Remainder distribution (one-by-one to clusters with available points)
            rem = budget_core - chosen.size
            if rem > 0:
                # Priority by remaining capacity (larger clusters first)
                capacities = [(c, max(0, cluster_to_indices[c].size - (base_q if cluster_to_indices[c].size >= base_q else cluster_to_indices[c].size))) for c in range(K)]
                capacities.sort(key=lambda x: -x[1])
                # Track for each cluster the next offset to pick
                picked_counts = {c: min(base_q, cluster_to_indices[c].size) for c in range(K)}
                for c, cap in capacities:
                    if rem <= 0:
                        break
                    if cap <= 0:
                        continue
                    idx = cluster_to_indices[c]
                    need_pos = picked_counts[c]
                    if need_pos >= idx.size:
                        continue
                    # compute distances if first time or reuse
                    diff = X_core[idx] - centers[c]
                    d2 = np.einsum('ij,ij->i', diff, diff)
                    order = np.argsort(d2)
                    chosen = np.concatenate([chosen, idx[order[need_pos:need_pos+1]].astype(np.int64)])
                    picked_counts[c] = need_pos + 1
                    rem -= 1
                    if rem <= 0:
                        break

            # Map back to global indices
            if chosen.size > 0:
                core_idx = core_candidates[chosen]

            # Top-up globally if still short (due to empty clusters)
            if core_idx.size < budget_core:
                core_idx = KMeansRepresentativeSelector.ensure_topup(core_idx, population_size=n, need=budget_core, random_state=self.random_state)

        # Overlap via random sampling
        overlap_idx = np.array([], dtype=np.int64)
        if budget_overlap > 0 and overlap_candidates.size > 0:
            rng = np.random.RandomState(self.random_state)
            take = min(budget_overlap, overlap_candidates.size)
            overlap_idx = rng.choice(overlap_candidates, size=take, replace=False)

        selected_idx = np.unique(np.concatenate([pin_idx, core_idx, overlap_idx]))
        return selected_idx, pin_idx, core_idx, overlap_idx
