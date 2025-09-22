# CIC-IDS2018 Clustering & Density-Aware Filtering Module (2_clustering)

This module handles clustering and density-aware filtering of UMAP embeddings produced by module `1_encode`. It provides two specialized approaches: general clustering for most labels and density-aware filtering specifically for the massive Benign class.

## Overview

The `2_clustering` module consists of 2 complementary tools:

1. **General Clustering** (`clustering.py`) - For all major labels except Benign
2. **Density-Aware Filtering** (`benign_filter.py`) - Specialized for Benign train data

## Purpose & Strategy

### Problem Addressed
- **Benign class**: ~7M samples requiring intelligent downsampling to ~1.3M while preserving diversity
- **Other major classes**: Need cluster centers for downstream boundary computation and test selection
- **Memory constraints**: Large embeddings require efficient batch processing

### Approach
- **Benign**: 3-stage density-aware filter (small clusters + edge points + downsampled cores)
- **Others**: Standard MiniBatchKMeans clustering with optional cluster assignment

### Why Two Different Strategies?

**Scale & Purpose Differences**:
- **Benign**: ~7M samples requiring intelligent downsampling to ~1.3M while preserving diversity and boundary information
- **Others**: 100k-600k samples needing cluster centers for downstream boundary computation and fixed-budget compression

**Strategic Focus**:
- **Benign**: Diversity preservation through density-aware filtering (rare patterns + edge emphasis)
- **Others**: Coverage guarantee through standard clustering (centers for cross-class distance computation)

**Resource Constraints**:
- **Benign**: Memory limitations require streaming 3-stage approach
- **Others**: Manageable sizes allow standard batch processing

Both tools expect UMAP embeddings with `z_*` columns and `RowId` for traceability.

## Key Actions & Roles Summary

### Benign Density-Aware Filtering (`benign_filter.py`)
**Actions**:
- Micro-cluster with MiniBatchKMeans (K≈1500)
- Keep ALL samples from small clusters (preserve rare patterns)
- Keep edge points per cluster (high percentile distance from center)
- Downsample large dense clusters with minimum quotas
- Output: `cic2018_benign_embedding_filtered.parquet` + centers + `cluster_id`

**Roles**:
- **Scale Reduction**: 7M → ~1.3M samples (make kNN/relative margin computation feasible)
- **Boundary Emphasis**: Prioritize edge regions for better class separation
- **Diversity Preservation**: Maintain representative coverage across embedding space
- **Pipeline Foundation**: Prepare clustered structure for boundary detection and coreset selection

### Other Labels Clustering (`clustering.py`)
**Actions**:
- Standard MiniBatchKMeans clustering (default K=1500)
- Save cluster centers for train data reuse
- Optional in-place `cluster_id` assignment to embedding parquet

**Roles**:
- **Boundary Optimization**: Enable fast same-class distance computation (local kNN per cluster) for relative margin in module 3
- **Train-Test Alignment**: Provide train centers template for consistent test selection strategy
- **Compression Support**: Supply cluster structure for core/edge selection in module 4
- **Cross-Class Distance**: Support boundary computation between different labels

### In-Place Updates & Artifacts
**Data Modifications**:
- Add `cluster_id` column to both Benign filtered and other label embeddings
- Save `cic2018_{label}_kmeans_centers.npy` files for train data

**Strategic Value**:
- **Performance**: Accelerate relative margin computation via local clustering
- **Consistency**: Ensure train-test alignment (test assigned to train centers)
- **Traceability**: Maintain RowId mapping throughout pipeline
- **Integration**: Enable seamless handoff to modules 3 (confusion handling) and 4 (compression)

## Module Components

### 1. `clustering.py`
**Purpose**: Fits MiniBatchKMeans on UMAP embeddings and saves cluster centers for reuse.

**Key Features**:
- Standard clustering for all major labels (except Benign)
- Configurable K clusters (default: 1500) and batch size
- Optional in-place cluster_id assignment to embedding parquet
- Centers saved for downstream boundary computation and test selection
- Memory-efficient float32 processing

**Usage**:

Run:
```bash
activate_your_env_if_needed
$ python -m cli.cic2018.major.2_clustering.clustering --label Infilteration --save-cluster-id --log-level DEBUG
2025-09-07 19:46:51 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
2025-09-07 19:46:51 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=97538, z_dims=24 | RowId=True
2025-09-07 19:46:51 - __main__ - INFO - Loaded embeddings: rows=97538, dims=24
2025-09-07 19:46:51 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=1500, batch_size=10000)
2025-09-07 19:46:58 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(1500, 24)
2025-09-07 19:46:58 - __main__ - INFO - Fitted centers: shape=(1500, 24)
2025-09-07 19:46:58 - __main__ - INFO - Saved centers -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_kmeans_centers.npy
2025-09-07 19:46:58 - __main__ - INFO - Wrote cluster_id in-place -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_infilteration_embedding.parquet
```

Inputs:
- Embeddings (default resolved by config):
  - Benign: `embeddings/<subset>/cic2018_benign_embedding.parquet`
  - Others: `embeddings/<subset>/cic2018_<label>_embedding.parquet`

Outputs:
- Centers: `embeddings/train/cic2018_<label>_kmeans_centers.npy` (default)
- Optional embeddings with `cluster_id` (in-place update when `--save-cluster-id` is used)

Notes:
- Centers are saved under `train/` by default to reuse across subsets.
- Casting to float32 and batched KMeans improve performance on large datasets.

**Input**: `{DATA_FOLDER}/embeddings/{subset}/cic2018_{label}_embedding.parquet`
**Output**: 
- `{DATA_FOLDER}/embeddings/train/cic2018_{label}_kmeans_centers.npy`
- Optional: `cluster_id` column added in-place to embedding parquet

### 2. `benign_filter.py`
**Purpose**: Density-aware filtering pipeline specifically designed for the massive Benign class.

**Key Features**:
- 3-stage intelligent filtering: small clusters + edge points + downsampled cores
- Reduces ~7M Benign samples to ~1.3M (18%) while preserving diversity
- Micro-clustering with MiniBatchKMeans (K≈1500)
- Configurable filtering parameters (percentiles, keep rates, minimum quotas)
- Saves both filtered embeddings and cluster centers

**Filtering Strategy**:
1. **Small Clusters**: Keep ALL samples from clusters below size threshold
2. **Edge Points**: Keep high-percentile distance points from each cluster
3. **Core Downsampling**: Randomly downsample remaining large dense clusters

**Usage**:

Run:
```bash
activate_your_env_if_needed
$ python -m cli.cic2018.major.2_clustering.benign_filter --log-level DEBUG
2025-09-07 19:29:19 - __main__ - DEBUG - Density-aware filter hyperparameters: kmeans_k=1500,     kmeans_batch=10000, small_cluster_frac=0.0004,     edge_percentile=95.0, large_keep_rate=0.015,     min_keep_per_large=50, seed=42
2025-09-07 19:29:19 - __main__ - INFO - Loading embeddings from /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding.parquet
2025-09-07 19:29:21 - __main__ - INFO - RowId present in input: True
2025-09-07 19:29:21 - __main__ - INFO - Filtering 7066590 benign embeddings
2025-09-07 19:29:21 - resampling.undersampling.density_aware - INFO - Using MiniBatchKMeans for micro-clustering
2025-09-07 19:29:21 - resampling.undersampling.density_aware - INFO - Fitting MiniBatchKMeans
dist->centroid: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 7066590/7066590 [00:46<00:00, 152505.18it/s]
keep small clusters: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:06<00:00, 235.25it/s]
2025-09-07 19:30:32 - resampling.undersampling.density_aware - DEBUG - Small clusters threshold=2827 | points kept: 847748
keep edge points: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [01:13<00:00, 20.52it/s]
2025-09-07 19:31:45 - resampling.undersampling.density_aware - DEBUG - Edge points kept at 95.0th percentile | newly added: 330750
downsample large clusters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:06<00:00, 248.17it/s]
2025-09-07 19:31:51 - resampling.undersampling.density_aware - DEBUG - Downsampled large clusters | newly added: 94151
2025-09-07 19:31:52 - __main__ - INFO - Final kept after density filter: 1272649 / 7066590 (18.01%)
2025-09-07 19:31:55 - __main__ - INFO - RowId present in output: True
2025-09-07 19:31:55 - __main__ - INFO - Saved KMeans centers -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_benign_kmeans_centers.npy (shape=(1500, 24))
2025-09-07 19:31:55 - __main__ - INFO - Saved filtered Benign embeddings -> /dis/DS/minhtq/CIC-2018/embeddings/train/cic2018_benign_embedding_filtered.parquet
```

Inputs (default):
- `embeddings/train/cic2018_benign_embedding.parquet`

Outputs:
- Filtered Benign embeddings: `..._embedding_filtered.parquet`
- Centers (default): `embeddings/train/cic2018_benign_kmeans_centers.npy`

Strategy summary:
- Micro-cluster with MiniBatchKMeans (K ≈ 1500)
- Keep all samples in small clusters
- Keep edge points (above percentile per cluster)
- Downsample large dense clusters with a minimum kept quota

**Input**: `{DATA_FOLDER}/embeddings/train/cic2018_benign_embedding.parquet`


**Output**:
- `{DATA_FOLDER}/embeddings/train/cic2018_benign_embedding_filtered.parquet`
- `{DATA_FOLDER}/embeddings/train/cic2018_benign_kmeans_centers.npy`

## Complete Pipeline Execution

### For Benign (train data):
```bash
# Apply density-aware filtering
python -m cli.cic2018.major.2_clustering.benign_filter
```

### For Other Major Labels (train data):
```bash
# Generate cluster centers for each label
python -m cli.cic2018.major.2_clustering.clustering --label "Infilteration" --save-cluster-id
python -m cli.cic2018.major.2_clustering.clustering --label "DDoS attacks-LOIC-HTTP" --save-cluster-id
python -m cli.cic2018.major.2_clustering.clustering --label "Bot" --save-cluster-id
# ... repeat for other major labels
```

## Data Flow

```
UMAP Embeddings (from 1_encode)
    ↓
┌─── Benign (~7M samples) ───┐    ┌─── Other Labels ───┐
│   benign_filter.py         │    │   clustering.py    │
│   • Micro-cluster (K=1500) │    │   • MiniBatchKMeans │
│   • Keep small clusters    │    │   • Save centers   │
│   • Keep edge points       │    │   • Optional IDs   │
│   • Downsample cores       │    └────────────────────┘
│   ↓                        │              ↓
│   Filtered (~1.3M samples) │         Cluster Centers
│   + Cluster Centers        │         + Optional IDs
└─────────────────────────────┘
```

## Key Features

### Intelligent Filtering (Benign)
- **Diversity Preservation**: Small clusters kept entirely to preserve rare patterns
- **Boundary Emphasis**: Edge points prioritized for better class separation
- **Controlled Downsampling**: Large clusters downsampled with minimum quotas
- **Configurable Strategy**: Tunable thresholds and percentiles

### Efficient Clustering (Others)
- **Batch Processing**: MiniBatchKMeans for memory efficiency
- **Reusable Centers**: Saved for downstream boundary computation
- **In-Place Updates**: Optional cluster_id assignment without file duplication
- **Standardized Outputs**: Consistent naming for pipeline integration

### Performance Optimization
- **Float32 Processing**: Memory-efficient computation
- **Progress Tracking**: Detailed logging and progress bars
- **Configurable Batching**: Tunable for different hardware constraints

## Default Parameters

### Clustering (clustering.py)
- K clusters: 1500
- Batch size: 10,000
- Random state: 42

### Density-Aware Filter (benign_filter.py)
- K micro-clusters: 1500
- Small cluster threshold: 0.04% of total samples
- Edge percentile: 95th percentile
- Large cluster keep rate: 1.5%
- Minimum keep per large cluster: 50

## Output Structure

```
{DATA_FOLDER}/embeddings/train/
├── cic2018_benign_embedding_filtered.parquet    # Filtered Benign (~1.3M samples)
├── cic2018_benign_kmeans_centers.npy           # Benign cluster centers
├── cic2018_infilteration_kmeans_centers.npy    # Infilteration centers
├── cic2018_ddos_attacks-loic-http_kmeans_centers.npy  # Other label centers
└── ...                                          # Additional label centers
```

## Requirements & Dependencies

- **Input**: UMAP embeddings from module `1_encode` with `z_*` columns and `RowId`
- **Configuration**: Paths defined in `configs.cic2018.py`
- **Dependencies**: `resampling.undersampling` classes for filtering algorithms
- **Memory**: ~8-16GB RAM recommended for Benign filtering
