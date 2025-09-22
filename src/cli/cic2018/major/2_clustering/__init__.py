"""
CIC-IDS2018 Clustering & Density-Aware Filtering Module (2_clustering)

2-tool pipeline for clustering UMAP embeddings:
1. clustering.py - Standard MiniBatchKMeans for most major labels
2. benign_filter.py - Specialized density-aware filtering for massive Benign class

Key features: intelligent downsampling, cluster center generation, in-place updates.
See README.md for detailed usage and filtering strategies.
"""

__all__ = [
    "clustering",
    "benign_filter"
]
