"""
CIC-IDS2018 Confusion Handling & Boundary Detection Module (3_confusion_handle)

3-tool pipeline for advanced boundary detection and coreset selection:
1. compute_boundary.py - Boundary scoring with relative margin and selection
2. coreset_train.py - Balanced coreset (PIN + core + overlap) for train data
3. select_test_ids.py - Multi-component test selection using train centers

Key features: relative margin boundary detection, in-place updates, train-test alignment.
See README.md for detailed usage and selection strategies.
"""

__all__ = [
    "compute_boundary",
    "coreset_train", 
    "select_test_ids"
]
