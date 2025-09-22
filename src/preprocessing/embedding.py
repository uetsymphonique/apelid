import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.decomposition import PCA

import umap 



def build_union_feature_matrix(file_paths: list[str], label_col: str = 'Label') -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fp in file_paths:
        df = pd.read_csv(fp, low_memory=False)
        X = df.drop(columns=[label_col], errors='ignore')
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def fit_or_load_pca(
    X_union: pd.DataFrame,
    mode: str,
    target_variance: float,
    max_or_fixed_components: int,
    seed: int,
    model_dir: str,
    save_models: bool,
    load_models: bool,
    model_name: str = 'pca_major.pkl',
) -> Tuple[PCA, int, float]:
    pca_path = os.path.join(model_dir, model_name)
    if load_models and os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        if hasattr(pca, 'n_components_'):
            pca_dims = int(pca.n_components_)
        else:
            pca_dims = int(pca.n_components)
        pca_var_retained = float(getattr(pca, 'explained_variance_ratio_', np.array([])).sum()) if hasattr(pca, 'explained_variance_ratio_') else np.nan
        return pca, pca_dims, pca_var_retained

    if mode == 'auto':
        pca_probe = PCA(n_components=float(target_variance), random_state=seed)
        pca_probe.fit(X_union)
        auto_dims = int(getattr(pca_probe, 'n_components_', max_or_fixed_components))
        pca_dims = min(auto_dims, int(max_or_fixed_components))
        pca = PCA(n_components=pca_dims, random_state=seed)
        pca.fit(X_union)
        pca_var_retained = float(getattr(pca, 'explained_variance_ratio_', np.array([])).sum())
    else:
        pca_dims = int(max_or_fixed_components)
        pca = PCA(n_components=pca_dims, random_state=seed)
        pca.fit(X_union)
        pca_var_retained = float(getattr(pca, 'explained_variance_ratio_', np.array([])).sum())

    if save_models:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pca, pca_path)
    return pca, pca_dims, pca_var_retained


def fit_or_load_umap(
    X_union_pca: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    model_dir: str,
    save_models: bool,
    load_models: bool,
    model_name: str = 'umap_major.pkl',
):
    umap_path = os.path.join(model_dir, model_name)
    if load_models and os.path.exists(umap_path):
        um = joblib.load(umap_path)
        umap_dims = int(getattr(um, 'n_components', n_components))
        return um, umap_dims

    umap_dims = int(n_components)
    um = umap.UMAP(
        n_components=umap_dims,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=seed,
    )
    um.fit(X_union_pca)
    if save_models:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(um, umap_path)
    return um, umap_dims


def make_metadata_frame(
    n_rows: int,
    label_id: int,
    label_name: str | None,
    label_safe: str,
    encoder_num: str,
    pca_dims: int,
    pca_var: float,
    umap_dims: int,
    umap_neighbors: int,
    umap_min_dist: float,
    metric: str,
    seed: int,
    source: str,
) -> pd.DataFrame:
    meta = {
        'Label': np.full(n_rows, label_id, dtype=np.int32),
        'LabelName': np.full(n_rows, label_name if label_name is not None else '', dtype=object),
        'LabelSafe': np.full(n_rows, label_safe, dtype=object),
        'EncoderNum': np.full(n_rows, encoder_num, dtype=object),
        'PCA_VarRetained': np.full(n_rows, np.float32(pca_var), dtype=np.float32),
        'PCA_n': np.full(n_rows, np.int16(pca_dims), dtype=np.int16),
        'UMAP_n': np.full(n_rows, np.int16(umap_dims), dtype=np.int16),
        'UMAP_neighbors': np.full(n_rows, np.int16(umap_neighbors), dtype=np.int16),
        'UMAP_min_dist': np.full(n_rows, np.float32(umap_min_dist), dtype=np.float32),
        'Metric': np.full(n_rows, metric, dtype=object),
        'Seed': np.full(n_rows, np.int32(seed), dtype=np.int32),
        'Source': np.full(n_rows, source, dtype=object),
    }
    return pd.DataFrame(meta)


def compute_embeddings(X: pd.DataFrame | np.ndarray, pca: PCA, um) -> np.ndarray:
    Z_pca = pca.transform(X)
    Z = um.transform(Z_pca).astype(np.float32)
    return Z


def save_embeddings_parquet(Z: np.ndarray, metadata_df: pd.DataFrame, out_path: str,
                            engine: str = 'pyarrow', compression: str = 'snappy') -> None:
    z_cols = [f"z_{i+1}" for i in range(Z.shape[1])]
    emb_df = pd.DataFrame(Z, columns=z_cols, index=metadata_df.index)
    out_df = pd.concat([emb_df, metadata_df], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path, engine=engine, compression=compression, index=False)


def load_embedding_models(model_dir: str,
                          pca_model_name: str = 'pca_major.pkl',
                          umap_model_name: str = 'umap_major.pkl'):
    """Load pre-fitted PCA and UMAP models; raise if missing."""
    pca_path = os.path.join(model_dir, pca_model_name)
    umap_path = os.path.join(model_dir, umap_model_name)
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Missing PCA model: {pca_path}")
    if not os.path.exists(umap_path):
        raise FileNotFoundError(f"Missing UMAP model: {umap_path}")
    pca = joblib.load(pca_path)
    um = joblib.load(umap_path)
    return pca, um


