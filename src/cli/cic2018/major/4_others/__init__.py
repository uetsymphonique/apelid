"""
Module 4_others: Simple compression for major classes (non-Benign/Infilteration)

Lightweight core + edge compression built on MiniBatchKMeans clusters for
major classes that do not require the heavy Benign–Infilteration pipeline.

Components:
- compression.py: core + edge selection with proportional edge allocation

Key features:
- In-place write of `cluster_id`, `dist_center` into the base embeddings parquet
- Batch mode to process all major labels (`--label All`)
- Standardized output naming for train/test

See README.md for details and examples.
"""


