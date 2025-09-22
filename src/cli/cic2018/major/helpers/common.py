import os
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def load_embeddings(path: str, float32: bool = True) -> pd.DataFrame:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.debug(f"Loading embeddings: {path}")
    if not os.path.exists(path):
        raise SystemExit(f"Embedding parquet not found: {path}")
    df = pd.read_parquet(path)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    if not z_cols:
        raise SystemExit(f"No embedding columns (z_*) in: {path}")
    if float32:
        df[z_cols] = df[z_cols].astype(np.float32, copy=False)
    logger.debug(f"Loaded embeddings: rows={len(df)}, z_dims={len(z_cols)} | RowId={'RowId' in df.columns}")
    return df


def batched_knn_distances(X_query: np.ndarray,
                          nn,  # sklearn.neighbors.NearestNeighbors
                          batch_size: int,
                          desc: str) -> np.ndarray:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.debug(f"kNN distances batched: batch_size={batch_size}, desc={desc}")
    n = len(X_query)
    dists = np.empty(n, dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        d, _ = nn.kneighbors(X_query[start:end], n_neighbors=1, return_distance=True)
        dists[start:end] = d.ravel().astype(np.float32, copy=False)
    return dists


