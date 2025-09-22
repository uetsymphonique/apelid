"""
Major pipeline for CIC-IDS2018:

1) 1_encode: encode → PCA cache → UMAP embeddings (with RowId)
2) 2_clustering: Benign density-aware filter; others: clustering + centers
3) 3_confusion_handle: boundary detection, coreset (train), test selection
4) 4_others: simple compression for other major classes
5) 5_complete: finalize selections back to encoded/raw_processed

See README.md for data flow, standardized outputs, and usage.
"""


