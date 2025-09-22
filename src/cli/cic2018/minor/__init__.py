"""
Minor classes pipeline (encode → augment → decode)

Components:
- encode.py: per-label encoding (one-hot + MinMax/Quantile), no PCA/UMAP
- augmenting.py: WGAN-based augmentation on encoded train (CFM/FDM placeholders)
- decode_augmented.py: inverse-transform augmented encoded back to raw_processed

Key features:
- Works per minority label from `configs.cic2018.MINORITY_LABELS`
- Uses pre-fitted encoders; avoids leakage (train-fitted encoders only)
- Standardized file naming for encoded and raw_processed outputs

See README.md for usage, data flow, and I/O conventions.
"""


