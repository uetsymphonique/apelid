# CIC-IDS2018 Confusion Handling & Boundary Detection Module (3_confusion_handle)

This module addresses the core challenge of **Benign ↔ Infilteration confusion** through advanced boundary detection and intelligent coreset selection. It transforms clustered embeddings from modules 1-2 into optimized training and test sets.

## Overview

The `3_confusion_handle` module consists of 3 sequential tools:

1. **Boundary Detection** (`compute_boundary.py`) - Advanced boundary scoring with relative margin
2. **Train Coreset Selection** (`coreset_train.py`) - Balanced coreset construction (PIN + core + overlap)
3. **Test Selection** (`select_test_ids.py`) - Multi-component test selection using train alignment

## Purpose & Strategy

### Problem Addressed
- **Benign ↔ Infilteration Confusion**: High misclassification between these classes due to overlapping feature distributions
- **Class Imbalance**: Need balanced representation while preserving boundary information
- **Train-Test Consistency**: Ensure test selection aligns with train boundary detection strategy

### Approach
- **Relative Margin**: Use `d_cross / d_same` ratio to identify true boundary points (not deep-inside-other)
- **PIN Strategy**: Prioritize boundary points (PIN = Points of Interest at Boundary) in coreset
- **3-Component Coreset**: Combine boundary (PIN) + core representatives + overlap quota
- **Train-Test Alignment**: Use train cluster centers to ensure consistent test selection

All scripts operate on UMAP embeddings with clustering metadata from module `2_clustering`, with paths resolved via `configs.cic2018.py`.

## Module Components

### 1. `compute_boundary.py`
**Purpose**: Advanced boundary detection using relative margin to identify true boundary points between Benign and Infilteration.

**Key Features**:
- **Relative Margin Calculation**: `d_cross / d_same` ratio to distinguish boundary vs deep-inside points
- **Cluster-Optimized kNN**: Uses `cluster_id` from module 2 for fast same-class distance computation
- **Margin Band Filtering**: Exclude deep-inside-other (< 0.5) and deep-inside-own (> 2.0) points
- **In-Place Updates**: Writes `boundary_score` and `role='boundary'` directly to embedding parquet
- **Reuse Capability**: Can reuse existing scores for re-selection with different budgets

**Usage**:

Compute cross-class proximity and (optionally) relative margin, then mark selected points as boundary.

Run (relative margin recommended):
```bash
$ python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Benign --neighbor Infilteration --score p95 --use-relative-margin --select --log-level DEBUG
2025-09-07 19:51:46 - __main__ - DEBUG - Boundary params: subset=train, target=Benign, neighbor=Infilteration, k=10, batch_size=100000, score=p95, use_relative_margin=True, margin_band=[0.5, 2.0], select=True, budget=7000, float32=False
2025-09-07 19:51:46 - __main__ - INFO - Target (Benign filtered): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-07 19:51:46 - __main__ - INFO - Neighbor (Infilteration): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-07 19:51:47 - __main__ - INFO - Neighbor Z: (97538, 24), Target Z: (1272649, 24), k=10, batch=100000
2025-09-07 19:52:30 - __main__ - INFO - Relative margin stats: p10=6.4262, p25=19.9975, p50=87.0913, p75=773.1920, p90=10167.4879, p95=22058.3351, p99=167167.2887
2025-09-07 19:52:30 - __main__ - INFO - Marked role='boundary' for 7000 rows (selected PIN)
2025-09-07 19:52:33 - __main__ - INFO - Updated target embedding in-place -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet (added 'boundary_score', updated role)

$ python -m cli.cic2018.major.3_confusion_handle.compute_boundary --neighbor Benign --target Infilteration --score min --use-relative-margin --select --log-level DEBUG
2025-09-07 19:57:31 - __main__ - DEBUG - Boundary params: subset=train, target=Infilteration, neighbor=Benign, k=10, batch_size=100000, score=min, use_relative_margin=True, margin_band=[0.5, 2.0], select=True, budget=7000, float32=False
2025-09-07 19:57:31 - __main__ - INFO - Target (Infilteration): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-07 19:57:31 - __main__ - INFO - Neighbor (Benign filtered): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-07 19:57:31 - __main__ - INFO - Neighbor Z: (1272649, 24), Target Z: (97538, 24), k=10, batch=100000
2025-09-07 19:57:55 - __main__ - INFO - Relative margin stats: p10=0.0000, p25=0.3952, p50=0.9717, p75=2.0263, p90=6.2748, p95=15.9973, p99=250.8716
2025-09-07 19:57:55 - __main__ - INFO - Marked role='boundary' for 7000 rows (selected PIN)
2025-09-07 19:57:55 - __main__ - INFO - Updated target embedding in-place -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet (added 'boundary_score', updated role)
```

Behavior:
- Always writes `boundary_score` into the target embedding parquet IN-PLACE.
- With `--select`, also sets `role='boundary'` for selected rows IN-PLACE.
- If `cluster_id` exists in target, same-class distances use local kNN per cluster; otherwise direct kNN.

Inputs (auto-resolved):
- Target: `embeddings/<subset>/cic2018_<label>_embedding[_filtered].parquet` (Benign uses filtered)
- Neighbor: same pattern

**Input**: UMAP embeddings with `cluster_id` from module 2
**Output**: In-place updates with `boundary_score` and `role='boundary'` columns

### 2. `coreset_train.py`
**Purpose**: Constructs balanced coresets using 3-component strategy prioritizing boundary points.

**Key Features**:
- **PIN-First Strategy**: Boundary points (from `compute_boundary.py`) are preserved first
- **3-Component Balance**: PIN (boundary) + Core (representatives) + Overlap (robustness)
- **Margin-Aware Core Selection**: Excludes deep-overlap regions for core representatives
- **Budget Distribution**: Automatic allocation across components with configurable ratios
- **Quality Control**: Coverage radius and boundary score statistics for validation

**Coreset Strategy**:

Build a balanced coreset using:
- PIN: boundary points (from `compute_boundary`) are kept first
- Core: KMeans representative selection (closest-to-centroid) on remaining clean regions
- Overlap: small random quota from deep-overlap region for robustness

Run:
```bash
]$ python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Infilteration --use-relative-margin --log-level DEBUG
2025-09-07 22:39:49 - __main__ - DEBUG - Hyperparameters: subset=train, label=Infilteration, budget_total=28000, kmeans_batch=10000, n_clusters=None, overlap_ratio=0.05, min_margin=2.0, use_relative_margin=True, qc_cover_sample=300000, qc_enable=True
2025-09-07 22:39:49 - __main__ - INFO - Using input embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-07 22:39:49 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-07 22:39:50 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=97538, z_dims=24 | RowId=True
2025-09-07 22:39:50 - __main__ - INFO - RowId present (input): True
2025-09-07 22:39:50 - __main__ - INFO - PIN from embeddings: 7000 points
2025-09-07 22:39:50 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=19950, batch_size=10000)
2025-09-07 23:57:39 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(19950, 24) in 4669.40 seconds
2025-09-07 23:57:40 - __main__ - INFO - Final coreset composition: PIN=7000, Core=19950, Overlap=1050, Total=28000
2025-09-07 23:57:40 - __main__ - DEBUG - Computing dist_to_S: distance to nearest PIN
dist other->PIN: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:17<00:00, 17.60s/it]
2025-09-07 23:57:59 - __main__ - INFO - Covering radius (nearest selected): p50=0.0140, p90=0.1547, p95=0.2465
2025-09-07 23:57:59 - __main__ - INFO - boundary_score in coreset [boundary-only]: mean=114.1711, p50=3.0962, p90=25.6429, p95=81.0860
2025-09-07 23:57:59 - __main__ - INFO - Role distribution: {'core': 19950, 'boundary': 7000, 'overlap': 1050}
2025-09-07 23:57:59 - __main__ - INFO - RowId present (output): True
2025-09-07 23:57:59 - __main__ - INFO - Saved balanced coreset -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding_compressed_coreset.parquet




$ python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Benign --use-relative-margin --log-level DEBUG
2025-09-08 01:04:53 - __main__ - DEBUG - Hyperparameters: subset=train, label=Benign, budget_total=28000, kmeans_batch=10000, n_clusters=None, overlap_ratio=0.05, min_margin=2.0, use_relative_margin=True, qc_cover_sample=300000, qc_enable=True
2025-09-08 01:04:53 - __main__ - INFO - Using input embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-08 01:04:53 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-08 01:04:54 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=1272649, z_dims=24 | RowId=True
2025-09-08 01:04:54 - __main__ - INFO - RowId present (input): True
2025-09-08 01:04:54 - __main__ - INFO - PIN from embeddings: 7000 points
2025-09-08 01:04:55 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=19950, batch_size=10000)
2025-09-08 01:09:03 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(19950, 24) in 248.33 seconds
2025-09-08 01:09:17 - __main__ - INFO - Final coreset composition: PIN=7000, Core=19950, Overlap=1050, Total=28000
2025-09-08 01:09:17 - __main__ - DEBUG - Computing dist_to_S: distance to nearest PIN
dist other->PIN: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:14<00:00, 14.88s/it]
2025-09-08 01:09:34 - __main__ - INFO - Covering radius (nearest selected): p50=0.0092, p90=0.0604, p95=0.0929
2025-09-08 01:09:34 - __main__ - INFO - boundary_score in coreset [boundary-only]: mean=4758.9932, p50=18.6533, p90=3114.7866, p95=10328.7740
2025-09-08 01:09:34 - __main__ - INFO - Role distribution: {'core': 19950, 'boundary': 7000, 'overlap': 1050}
2025-09-08 01:09:34 - __main__ - INFO - RowId present (output): True
2025-09-08 01:09:34 - __main__ - INFO - Saved balanced coreset -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered_compressed_coreset.parquet


2025-09-15 08:22:07 - __main__ - DEBUG - Boundary params: subset=train, target=Benign, neighbor=Infilteration, k=10, batch_size=100000, score=min, use_relative_margin=True, margin_band=[0.9, 1.2], select=True, budget=700, float32=True
2025-09-15 08:22:07 - __main__ - INFO - Target (Benign filtered): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-15 08:22:07 - __main__ - INFO - Neighbor (Infilteration): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-15 08:22:08 - __main__ - INFO - Neighbor Z: (52960, 24), Target Z: (770357, 24), k=10, batch=100000
2025-09-15 08:22:33 - __main__ - INFO - Relative margin stats: p10=1.4422, p25=2.8824, p50=9.6229, p75=93.8811, p90=854.8208, p95=9743.1545, p99=30408.6204
2025-09-15 08:22:33 - __main__ - INFO - Marked role='boundary' for 700 rows (selected PIN)
2025-09-15 08:22:35 - __main__ - INFO - Updated target embedding in-place -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet (added 'boundary_score', updated role)
2025-09-15 08:22:37 - __main__ - INFO - Using input embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-15 08:22:37 - __main__ - INFO - RowId present (input): True
2025-09-15 08:22:37 - __main__ - INFO - PIN from embeddings: 700 points
2025-09-15 08:23:40 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12635, 24) in 62.63 seconds
2025-09-15 08:23:45 - __main__ - INFO - Final coreset composition: PIN=700, Core=12635, Overlap=665, Total=14000
dist other->PIN: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.07it/s]
2025-09-15 08:23:47 - __main__ - INFO - Covering radius (nearest selected): p50=0.0138, p90=0.0819, p95=0.1196
2025-09-15 08:23:47 - __main__ - INFO - boundary_score in coreset [boundary-only]: mean=701.6780, p50=10.8267, p90=425.1867, p95=1227.6747
2025-09-15 08:23:47 - __main__ - INFO - Role distribution: {'core': 12635, 'boundary': 700, 'overlap': 665}
2025-09-15 08:23:47 - __main__ - INFO - RowId present (output): True
2025-09-15 08:23:47 - __main__ - INFO - Saved balanced coreset -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered_compressed_coreset.parquet
2025-09-15 08:23:48 - __main__ - DEBUG - Boundary params: subset=train, target=Infilteration, neighbor=Benign, k=10, batch_size=100000, score=min, use_relative_margin=True, margin_band=[0.7, 1.35], select=True, budget=12500, float32=True
2025-09-15 08:23:48 - __main__ - INFO - Target (Infilteration): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-15 08:23:48 - __main__ - INFO - Neighbor (Benign filtered): /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_embedding_filtered.parquet
2025-09-15 08:23:49 - __main__ - INFO - Neighbor Z: (770357, 24), Target Z: (52960, 24), k=10, batch=100000
2025-09-15 08:24:01 - __main__ - INFO - Relative margin stats: p10=0.1000, p25=0.4359, p50=0.8880, p75=1.4895, p90=3.3015, p95=6.5577, p99=158.8966
2025-09-15 08:24:01 - __main__ - INFO - Marked role='boundary' for 12500 rows (selected PIN)
2025-09-15 08:24:01 - __main__ - INFO - Updated target embedding in-place -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet (added 'boundary_score', updated role)
2025-09-15 08:24:03 - __main__ - INFO - Using input embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-15 08:24:03 - __main__ - INFO - RowId present (input): True
2025-09-15 08:24:03 - __main__ - INFO - PIN from embeddings: 12500 points
2025-09-15 08:24:06 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(1425, 24) in 2.48 seconds
2025-09-15 08:24:06 - __main__ - INFO - Final coreset composition: PIN=12500, Core=1425, Overlap=75, Total=14000
dist other->PIN: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.60s/it]
2025-09-15 08:24:08 - __main__ - INFO - Covering radius (nearest selected): p50=0.0081, p90=0.1189, p95=0.2015
2025-09-15 08:24:08 - __main__ - INFO - boundary_score in coreset [boundary-only]: mean=17.9869, p50=0.9979, p90=2.0264, p95=3.3040
2025-09-15 08:24:08 - __main__ - INFO - Role distribution: {'boundary': 12500, 'core': 1425, 'overlap': 75}
2025-09-15 08:24:08 - __main__ - INFO - RowId present (output): True
2025-09-15 08:24:08 - __main__ - INFO - Saved balanced coreset -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding_compressed_coreset.parquet
```

Requirements:
- Target embedding must include `boundary_score` (and optionally `role='boundary'`) written in-place by `compute_boundary`.
- Output: `<...>_compressed_coreset.parquet` with metadata: `role` (boundary/core/overlap), `dist_to_S`.

**Input**: Embeddings with `boundary_score` and `role` from `compute_boundary.py`
**Output**: `{label}_embedding_compressed_coreset.parquet` with 28k balanced samples

### 3. `select_test_ids.py`
**Purpose**: Multi-component test selection using train cluster centers for alignment.

**Key Features**:
- **Train-Test Alignment**: Uses train cluster centers from module 2 for consistent assignment
- **4-Component Strategy**: Small clusters + Edge points + Boundary points + Core sampling
- **Adaptive Parameters**: Different strategies for Benign (0.4% compression) vs Infilteration (30% compression)
- **Cross-Class Boundary**: Computes boundary scores between test target and neighbor
- **In-Place Annotation**: Adds cluster assignment and boundary information to test embeddings

**Selection Strategy**:

Select representative test rows via stratified strategy:
- Assign clusters using centers from train
- Keep: small clusters (100%), edges (top percentile), boundary subset (quota), and core subsample stratified by cluster

Run (Benign):
```bash
$ python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Benign --log-level DEBUG

2025-09-08 01:09:36 - __main__ - DEBUG - Select test ids hyperparameters: label=Benign, budget_total=12000, small_cluster_frac=None, edge_percentile=95.0, boundary_frac=None, large_rate=None, min_quota=8, knn_batch=200000
2025-09-08 01:09:36 - __main__ - INFO - Benign strategy: small_frac=0.0003, boundary_frac=0.12, large_rate=0.003
2025-09-08 01:09:37 - __main__ - INFO - Loaded Benign_test=3028537 infilteration_test=41803 | RowId present: target=True
2025-09-08 01:09:37 - __main__ - INFO - Loading centers: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_kmeans_centers.npy
2025-09-08 01:09:37 - __main__ - INFO - Centers shape: (1500, 24)
assign clusters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:01<00:00, 10.84it/s]
2025-09-08 01:09:39 - __main__ - INFO - dist_center: min=0.0002, mean=0.1852, p50=0.1235, p90=0.2978, p95=0.4025, max=7.2480
2025-09-08 01:09:39 - __main__ - INFO - Small clusters threshold=909 | num_small_clusters=346 | points_in_small=200677
edge per cluster: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:02<00:00, 735.92it/s]
2025-09-08 01:09:41 - __main__ - INFO - Edge points kept at p95.0: total=152119
knn cross-class: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:21<00:00,  1.35s/it]
2025-09-08 01:10:03 - __main__ - INFO - Boundary quota=1440 | cutoff boundary_score=1.000000
2025-09-08 01:10:03 - __main__ - INFO - Selected after boundary+small+edge: 343687 (boundary=1440, small=200677, edge=152119)
2025-09-08 01:10:04 - __main__ - INFO - Trimmed edge-only by 141814, remaining_excess=189873
2025-09-08 01:21:04 - __main__ - INFO - Trimmed small-only by 189873
2025-09-08 01:21:04 - __main__ - INFO - Selected after trim = 12000 | target=12000
2025-09-08 01:21:11 - __main__ - INFO - Wrote in-place to test parquet -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_benign_embedding.parquet (added 'cluster_id', 'dist_center', 'boundary_score', 'role')
2025-09-08 01:21:11 - __main__ - INFO - Final selected = 12000 / 12000 | boundary=1440, small=10804, edge=669, core=-913
2025-09-08 01:21:11 - __main__ - INFO - Saved selected RowIds -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_benign_test_selected_rowids.parquet (rows=12000) | role_breakdown={'small': 10246, 'boundary': 1085, 'edge': 669}

$ python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Benign --log-level DEBUG
2025-09-15 13:54:43 - __main__ - DEBUG - Select test ids hyperparameters: label=Benign, budget_total=6000, small_cluster_frac=None, edge_percentile=95.0, boundary_frac=None, large_rate=None, min_quota=8, knn_batch=200000
2025-09-15 13:54:43 - __main__ - INFO - Benign strategy: small_frac=0.0003, boundary_frac=0.12, large_rate=0.003
2025-09-15 13:54:44 - __main__ - INFO - Loaded Benign_test=1977313 infilteration_test=22698 | RowId present: target=True
2025-09-15 13:54:44 - __main__ - INFO - Loading centers: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_kmeans_centers.npy
2025-09-15 13:54:44 - __main__ - INFO - Centers shape: (1500, 24)
assign clusters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 10.15it/s]
2025-09-15 13:54:46 - __main__ - INFO - dist_center: min=0.0001, mean=0.1375, p50=0.0991, p90=0.2030, p95=0.2556, max=6.3058
2025-09-15 13:54:46 - __main__ - INFO - Small clusters threshold=594 | num_small_clusters=302 | points_in_small=105632
edge per cluster: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:01<00:00, 936.35it/s]
2025-09-15 13:54:47 - __main__ - INFO - Edge points kept at p95.0: total=99583
knn cross-class: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:08<00:00,  1.16it/s]
2025-09-15 13:54:56 - __main__ - INFO - Boundary quota=720 | cutoff boundary_score=1.000000
2025-09-15 13:54:56 - __main__ - INFO - Selected after boundary+small+edge: 200285 (boundary=720, small=105632, edge=99583)
2025-09-15 13:54:57 - __main__ - INFO - Trimmed edge-only by 94103, remaining_excess=100182
2025-09-15 13:58:36 - __main__ - INFO - Trimmed small-only by 100182
2025-09-15 13:58:36 - __main__ - INFO - Selected after trim = 6000 | target=6000
2025-09-15 13:58:41 - __main__ - INFO - Wrote in-place to test parquet -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_benign_embedding.parquet (added 'cluster_id', 'dist_center', 'boundary_score', 'role')
2025-09-15 13:58:41 - __main__ - INFO - Final selected = 6000 / 6000 | boundary=720, small=5450, edge=367, core=-537
2025-09-15 13:58:41 - __main__ - INFO - Saved selected RowIds -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_benign_test_selected_rowids.parquet (rows=6000) | role_breakdown={'small': 5132, 'boundary': 501, 'edge': 367}
```

Run (Infilteration):
```bash
$ python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Infilteration --log-level DEBUG
2025-09-08 01:21:13 - __main__ - DEBUG - Select test ids hyperparameters: label=Infilteration, budget_total=12000, small_cluster_frac=None, edge_percentile=95.0, boundary_frac=None, large_rate=None, min_quota=8, knn_batch=200000
2025-09-08 01:21:13 - __main__ - INFO - Infilteration strategy: small_frac=0.01, boundary_frac=0.1, large_rate=0.25
2025-09-08 01:21:14 - __main__ - INFO - Loaded Infilteration_test=41803 benign_test=3028537 | RowId present: target=True
2025-09-08 01:21:14 - __main__ - INFO - Loading centers: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_infilteration_kmeans_centers.npy
2025-09-08 01:21:14 - __main__ - INFO - Centers shape: (1500, 24)
assign clusters: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.76it/s]
2025-09-08 01:21:14 - __main__ - INFO - dist_center: min=0.0001, mean=0.1748, p50=0.1266, p90=0.2830, p95=0.3582, max=6.6726
2025-09-08 01:21:14 - __main__ - INFO - Small clusters threshold=419 | num_small_clusters=1493 | points_in_small=41803
edge per cluster: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1493/1493 [00:00<00:00, 5125.44it/s]
2025-09-08 01:21:15 - __main__ - INFO - Edge points kept at p95.0: total=2829
knn cross-class: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:27<00:00, 27.28s/it]
2025-09-08 01:21:42 - __main__ - INFO - Boundary quota=1200 | cutoff boundary_score=1.000000
2025-09-08 01:21:42 - __main__ - INFO - Selected after boundary+small+edge: 41803 (boundary=1200, small=41803, edge=2829)
2025-09-08 01:21:42 - __main__ - INFO - Trimmed small-only by 29803
2025-09-08 01:21:42 - __main__ - INFO - Selected after trim = 12000 | target=12000
2025-09-08 01:21:43 - __main__ - INFO - Wrote in-place to test parquet -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_infilteration_embedding.parquet (added 'cluster_id', 'dist_center', 'boundary_score', 'role')
2025-09-08 01:21:43 - __main__ - INFO - Final selected = 12000 / 12000 | boundary=1200, small=12000, edge=876, core=-2076
2025-09-08 01:21:43 - __main__ - INFO - Saved selected RowIds -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_infilteration_test_selected_rowids.parquet (rows=12000) | role_breakdown={'small': 11124, 'edge': 876}

$ python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Infilteration --log-level DEBUG
2025-09-15 13:59:25 - __main__ - DEBUG - Select test ids hyperparameters: label=Infilteration, budget_total=6000, small_cluster_frac=None, edge_percentile=95.0, boundary_frac=None, large_rate=None, min_quota=8, knn_batch=200000
2025-09-15 13:59:25 - __main__ - INFO - Infilteration strategy: small_frac=0.01, boundary_frac=0.1, large_rate=0.25
2025-09-15 13:59:26 - __main__ - INFO - Loaded Infilteration_test=22698 benign_test=1977313 | RowId present: target=True
2025-09-15 13:59:26 - __main__ - INFO - Loading centers: /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_infilteration_kmeans_centers.npy
2025-09-15 13:59:26 - __main__ - INFO - Centers shape: (1500, 24)
assign clusters: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.05it/s]
2025-09-15 13:59:26 - __main__ - INFO - dist_center: min=0.0001, mean=0.0974, p50=0.0657, p90=0.1752, p95=0.2210, max=6.1933
2025-09-15 13:59:26 - __main__ - INFO - Small clusters threshold=227 | num_small_clusters=1477 | points_in_small=22698
edge per cluster: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1477/1477 [00:00<00:00, 6214.63it/s]
2025-09-15 13:59:27 - __main__ - INFO - Edge points kept at p95.0: total=1899
knn cross-class: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.40s/it]
2025-09-15 13:59:37 - __main__ - INFO - Boundary quota=600 | cutoff boundary_score=1.000000
2025-09-15 13:59:37 - __main__ - INFO - Selected after boundary+small+edge: 22698 (boundary=600, small=22698, edge=1899)
2025-09-15 13:59:37 - __main__ - INFO - Trimmed small-only by 16698
2025-09-15 13:59:37 - __main__ - INFO - Selected after trim = 6000 | target=6000
2025-09-15 13:59:38 - __main__ - INFO - Wrote in-place to test parquet -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_infilteration_embedding.parquet (added 'cluster_id', 'dist_center', 'boundary_score', 'role')
2025-09-15 13:59:38 - __main__ - INFO - Final selected = 6000 / 6000 | boundary=600, small=6000, edge=494, core=-1094
2025-09-15 13:59:38 - __main__ - INFO - Saved selected RowIds -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_infilteration_test_selected_rowids.parquet (rows=6000) | role_breakdown={'small': 5506, 'edge': 494}
```

Outputs:
- Parquet of selected `RowId` with basic metadata (`cluster_id`, `dist_center`, `boundary_score`, `role`).

**Input**: Test embeddings + train cluster centers from module 2
**Output**: 
- In-place test embedding updates with `cluster_id`, `dist_center`, `boundary_score`, `role`
- `{label}_test_selected_rowids.parquet` with 12k selected samples

## Complete Pipeline Execution

```bash
# Step 1: Compute boundary scores for both directions
python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Benign --neighbor Infilteration --use-relative-margin --select
python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Infilteration --neighbor Benign --use-relative-margin --select

# Step 2: Build balanced coresets for train
python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Benign --use-relative-margin
python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Infilteration --use-relative-margin

# Step 3: Select representative test samples
python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Benign
python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Infilteration
```

## Data Flow

```
UMAP Embeddings + cluster_id (from modules 1-2)
    ↓
compute_boundary.py → boundary_score + role='boundary' (in-place)
    ↓
coreset_train.py → 28k balanced coreset (PIN + core + overlap)
    ↓
select_test_ids.py → 12k test selection (train-aligned)
```

## Key Features

### Advanced Boundary Detection
- **Relative Margin**: Distinguishes true boundary from deep-inside points
- **Cluster Optimization**: Fast same-class kNN using module 2 clustering
- **Margin Band**: Intelligent filtering of boundary candidates
- **Bidirectional**: Computes boundaries from both class perspectives

### Intelligent Coreset Construction
- **PIN Priority**: Boundary points preserved first (most informative)
- **Core Representatives**: KMeans-based coverage of clean regions
- **Overlap Robustness**: Small quota of challenging overlap cases
- **Quality Control**: Coverage and boundary score validation

### Train-Test Consistency
- **Cluster Alignment**: Test uses train cluster centers for assignment
- **Strategy Adaptation**: Different parameters for different class scales
- **Multi-Component**: Balanced representation across pattern types
- **Boundary Emphasis**: Consistent boundary focus across train/test

### Performance & Scalability
- **In-Place Updates**: Minimize file proliferation and I/O overhead
- **Batch Processing**: Handle large embeddings efficiently
- **Memory Optimization**: Float32 and configurable batch sizes
- **Progress Tracking**: Detailed logging and progress bars

## Default Parameters

### Boundary Detection
- kNN neighbors: 10
- Margin band: [0.5, 2.0]
- Boundary budget: 7,000 per class
- Batch size: 100,000

### Coreset Construction
- Total budget: 28,000 per class
- PIN: ~7,000 (boundary points)
- Core: ~19,950 (representatives)
- Overlap: ~1,050 (5% robustness quota)

### Test Selection
- Total budget: 12,000 per class
- Benign: small_frac=0.0003, boundary_frac=0.12, large_rate=0.003
- Infilteration: small_frac=0.01, boundary_frac=0.10, large_rate=0.25

## Output Structure

```
{DATA_FOLDER}/embeddings/train/
├── cic2018_benign_embedding_filtered.parquet         # Updated with boundary_score, role
├── cic2018_infilteration_embedding.parquet           # Updated with boundary_score, role
├── cic2018_benign_embedding_filtered_compressed_coreset.parquet    # 14k train coreset
└── cic2018_infilteration_embedding_compressed_coreset.parquet      # 14k train coreset

{DATA_FOLDER}/embeddings/test/
├── cic2018_benign_embedding.parquet                  # Updated with cluster_id, boundary_score, role
├── cic2018_infilteration_embedding.parquet           # Updated with cluster_id, boundary_score, role
├── cic2018_benign_test_selected_rowids.parquet      # 6k test selection
└── cic2018_infilteration_test_selected_rowids.parquet # 6k test selection
```

## Requirements & Dependencies

- **Input**: UMAP embeddings from module 1_encode with `cluster_id` from module 2_clustering
- **Configuration**: Paths defined in `configs/cic2018.py`
- **Dependencies**: `resampling.undersampling` classes for boundary detection and coreset construction
- **Memory**: ~16-32GB RAM recommended for full pipeline execution

## Notes

- **In-Place Policy**: Intermediate annotations written directly to base embedding parquets
- **Performance**: Use `--float32` and batching flags for large datasets
- **Reproducibility**: Sensible defaults with tunable ratios/thresholds for different aggressiveness levels
- **Modularity**: Each component can be run independently with appropriate inputs
