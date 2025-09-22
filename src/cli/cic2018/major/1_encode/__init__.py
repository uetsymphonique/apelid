"""
CIC-IDS2018 Major Labels Encoding & Embedding Pipeline (Module 1_encode)

5-step pipeline for transforming clean merged data into embeddings:
1. encode.py - Apply pre-fitted encoders (LabelEncoder, OneHotEncoder, MinMaxScaler)
2. pca_fit.py - Fit PCA on train data (leakage-free)
3. pca_transform.py - Transform to PCA space with parquet caching
4. umap_fit.py - Fit UMAP with class-capped sampling
5. umap_transform.py - Generate final UMAP embeddings

Key features: leakage prevention, memory optimization, RowId traceability.
See README.md for detailed usage and configuration.
"""

__all__ = [
    "encode",
    "pca_fit", 
    "pca_transform",
    "umap_fit",
    "umap_transform"
]