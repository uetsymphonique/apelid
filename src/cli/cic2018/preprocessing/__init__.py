"""
CIC-IDS2018 Preprocessing Module

3-step preprocessing pipeline for CIC-IDS2018 dataset:
1. merge_clean.py - Raw CSV merging with early cleaning and per-label splitting
2. split_clean_merged.py - Train/test splitting to prevent data leakage
3. setup_encoders.py - Encoder fitting using training data only

Key features: schema harmonization, sentinel handling, leakage prevention.
See README.md for detailed usage and configuration.
"""

__all__ = [
    "merge_clean",
    "split_clean_merged", 
    "setup_encoders"
]
