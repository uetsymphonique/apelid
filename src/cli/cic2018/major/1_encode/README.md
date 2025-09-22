# CIC-IDS2018 Major Labels Encoding & Embedding Module (1_encode)

This module handles the complete encoding and embedding pipeline for majority labels (Benign, Infilteration, etc.) in the CIC-IDS2018 dataset. It transforms clean merged data through encoding, PCA, and UMAP to produce stable, reusable embeddings for downstream compression and analysis.

## Overview

The `1_encode` module consists of 5 sequential steps:

1. **Data Encoding** (`encode.py`)
2. **PCA Model Fitting** (`pca_fit.py`)
3. **PCA Transformation** (`pca_transform.py`)
4. **UMAP Model Fitting** (`umap_fit.py`)
5. **UMAP Transformation** (`umap_transform.py`)

## Rationale for PCA + UMAP Pipeline

The dual dimensionality reduction approach is specifically designed to address the **Benign ↔ Infilteration confusion problem**:

- **PCA (32 dimensions)**: Linear dimensionality reduction that preserves global variance structure while reducing computational complexity from ~70 encoded features to 32 components
- **UMAP (24 dimensions)**: Non-linear manifold learning that enhances cluster separation and preserves local neighborhood structure

**Expected Benefits**:
- **Improved Class Separation**: UMAP's non-linear transformation creates clearer boundaries between Benign and Infilteration classes
- **Efficient Distance Computation**: Cross-class distance calculations in 24D UMAP space are more computationally efficient and interpretable than in original 70D space
- **Enhanced Boundary Detection**: The embedding space facilitates more accurate identification of boundary points between confusing classes
- **Compact Representation**: Reduced dimensionality enables faster downstream processing (clustering, kNN, compression) while maintaining discriminative power

## Module Components

### 1. `encode.py`
**Purpose**: Applies pre-fitted encoders (LabelEncoder, OneHotEncoder, MinMaxScaler) to per-label clean merged CSVs.

**Key Features**:
- Uses ONLY pre-fitted encoders (fitted on train data) to prevent leakage
- Handles sentinel -1 values by adding `<feature>_is_missing` indicators and imputing
- Supports both train/test subsets or full mode (processes both)
- Saves both encoded and raw_processed versions with RowId preservation
- Comprehensive deduplication with logging

**Usage**:
```bash
# Encode train data for all major labels
python -m cli.cic2018.major.1_encode.encode --subset train

# Encode test data
python -m cli.cic2018.major.1_encode.encode --subset test

# Encode both train and test (full mode)
python -m cli.cic2018.major.1_encode.encode --subset full

# Encode specific labels only
python -m cli.cic2018.major.1_encode.encode --mode label --labels "Benign" "Infilteration"

# Custom sentinel imputation
python -m cli.cic2018.major.1_encode.encode --sentinel-impute median
```

**Input**: `{CLEAN_MERGED_DATA_FOLDER}/{subset}/cic2018_{label}_*_clean_merged.csv`
**Output**: 
- `{ENCODED_DATA_FOLDER}/{subset}/cic2018_{label}_encoded.csv`
- `{RAW_PROCESSED_DATA_FOLDER}/{subset}/cic2018_{label}_raw_processed.csv`

### 2. `pca_fit.py`
**Purpose**: Fits IncrementalPCA on the UNION of encoded TRAIN data across majority labels.

**Key Features**:
- Uses ONLY train data to prevent leakage
- Drops `*_is_missing` columns before fitting
- IncrementalPCA for memory efficiency with large datasets
- Configurable components and batch size
- Saves fitted model for reuse

**Usage**:
```bash
# Fit PCA with default settings (32 components)
python -m cli.cic2018.major.1_encode.pca_fit

# Custom PCA components
python -m cli.cic2018.major.1_encode.pca_fit --pca-components 64

# Adjust batch size for memory constraints
python -m cli.cic2018.major.1_encode.pca_fit --ipca-batch-size 50000

# Use float32 for memory optimization
python -m cli.cic2018.major.1_encode.pca_fit --float32
```

**Input**: `{ENCODED_DATA_FOLDER}/train/cic2018_{label}_encoded.csv`
**Output**: `{ENCODERS_FOLDER}/pca_major.pkl`

### 3. `pca_transform.py`
**Purpose**: Transforms encoded data to PCA space in chunks and caches results as parquet files.

**Key Features**:
- Streams large CSV files in chunks for memory efficiency
- Drops `*_is_missing` columns before transformation
- Saves per-label PCA cache in organized subdirectories
- Preserves RowId for exact mapping to original data
- Supports both train and test subsets

**Usage**:
```bash
# Transform train data to PCA space
python -m cli.cic2018.major.1_encode.pca_transform --subset train

# Transform test data
python -m cli.cic2018.major.1_encode.pca_transform --subset test

# Custom chunk size for memory tuning
python -m cli.cic2018.major.1_encode.pca_transform --read-chunk-rows 100000

# Use float32 precision
python -m cli.cic2018.major.1_encode.pca_transform --float32
```

**Input**: `{ENCODED_DATA_FOLDER}/{subset}/cic2018_{label}_encoded.csv`
**Output**: `{PCA_CACHE_FOLDER}/cache_pca_{subset}/{label_safe}/pca_*.parquet`

### 4. `umap_fit.py`
**Purpose**: Fits UMAP model on PCA-transformed TRAIN data with class-capped sampling.

**Key Features**:
- Class-capped streaming (Benign ≤ 200k, others ≤ 100k) to prevent OOM
- Reads directly from PCA cache for efficiency
- Configurable UMAP hyperparameters (components, neighbors, min_dist)
- Performance tuning options (epochs, negative sampling, low memory)
- NUMBA thread control for reproducibility

**Usage**:
```bash
# Fit UMAP with default settings (24 components)
python -m cli.cic2018.major.1_encode.umap_fit

# Custom UMAP parameters
python -m cli.cic2018.major.1_encode.umap_fit --umap-components 32 --umap-n-neighbors 15

# Performance tuning for faster fitting
python -m cli.cic2018.major.1_encode.umap_fit --n-epochs 100 --neg-sample-rate 1 --low-memory

# Full mode (no class capping - use with caution)
python -m cli.cic2018.major.1_encode.umap_fit --fit-mode full

# Control NUMBA threads
python -m cli.cic2018.major.1_encode.umap_fit --numba-threads 4
```

**Input**: `{PCA_CACHE_FOLDER}/cache_pca_train/{label_safe}/pca_*.parquet`
**Output**: `{ENCODERS_FOLDER}/umap_major.pkl`

### 5. `umap_transform.py`
**Purpose**: Transforms PCA data to UMAP embeddings per label, preserving RowId.

**Key Features**:
- Reads from PCA cache only (no direct CSV access)
- Processes each label independently for modularity
- Preserves RowId for downstream mapping
- Saves embeddings in standardized parquet format
- Supports both train and test subsets

**Usage**:
```bash
# Transform train data to UMAP embeddings
python -m cli.cic2018.major.1_encode.umap_transform --subset train

# Transform test data
python -m cli.cic2018.major.1_encode.umap_transform --subset test

# Process specific labels only
python -m cli.cic2018.major.1_encode.umap_transform --mode label --labels "Benign"

# Use float32 precision
python -m cli.cic2018.major.1_encode.umap_transform --float32
```

**Input**: `{PCA_CACHE_FOLDER}/cache_pca_{subset}/{label_safe}/pca_*.parquet`
**Output**: `{DATA_FOLDER}/embeddings/{subset}/cic2018_{label}_embedding*.parquet`

## Complete Pipeline Execution

```bash
# Step 1: Encode data (both train and test)
python -m cli.cic2018.major.1_encode.encode --subset full

# Step 2: Fit PCA on train data only
python -m cli.cic2018.major.1_encode.pca_fit

# Step 3: Transform to PCA space (train and test)
python -m cli.cic2018.major.1_encode.pca_transform --subset train
python -m cli.cic2018.major.1_encode.pca_transform --subset test

# Step 4: Fit UMAP on train PCA data only
python -m cli.cic2018.major.1_encode.umap_fit

# Step 5: Transform to UMAP embeddings (train and test)
python -m cli.cic2018.major.1_encode.umap_transform --subset train
python -m cli.cic2018.major.1_encode.umap_transform --subset test
```

## Data Flow

```
Clean Merged CSVs
    ↓ encode.py
Encoded CSVs + Raw Processed CSVs (with RowId)
    ↓ pca_fit.py (train only)
Fitted PCA Model
    ↓ pca_transform.py
PCA Cache Parquet (per-label, with RowId)
    ↓ umap_fit.py (train only)
Fitted UMAP Model
    ↓ umap_transform.py
UMAP Embeddings Parquet (per-label, with RowId)
```

## Key Features

### Data Leakage Prevention
- **Strict Train-Only Fitting**: All models (encoders, PCA, UMAP) are fitted using train data only
- **Test Transformation Only**: Test data is never used for fitting, only transformed using pre-fitted models

### Memory Optimization
- **Streaming Processing**: Large CSV files processed in chunks
- **Float32 Precision**: Optional float32 casting for memory efficiency
- **Incremental Algorithms**: IncrementalPCA for large datasets
- **Class-Capped Sampling**: UMAP fitting with balanced class sampling

### Performance Features
- **PCA Caching**: Intermediate PCA results cached as parquet for fast access
- **Per-Label Processing**: Independent processing enables parallelization
- **Configurable Batch Sizes**: Tunable for different memory constraints
- **NUMBA Control**: Thread management for reproducible results

### Traceability
- **RowId Preservation**: End-to-end RowId tracking for exact back-mapping
- **Comprehensive Logging**: Detailed progress and hyperparameter logging
- **Standardized Naming**: Consistent file naming conventions

## Configuration

Key configuration parameters in `configs.cic2018.py`:
- `MAJORITY_LABELS`: Labels processed by this module
- `ENCODED_DATA_FOLDER`: Encoded CSV output location
- `PCA_CACHE_FOLDER`: PCA parquet cache location
- `ENCODERS_FOLDER`: Model persistence location

## Default Hyperparameters

### PCA
- Components: 32
- Batch Size: 100,000
- Chunk Rows: 200,000

### UMAP
- Components: 24
- Neighbors: 25
- Min Distance: 0.25
- Epochs: 200
- Negative Sample Rate: 2

### Class Caps (UMAP Fit)
- Benign: ≤ 200,000 samples
- Other Labels: ≤ 100,000 samples

## Output Structure

```
{DATA_FOLDER}/
├── encoded/
│   ├── train/cic2018_{label}_encoded.csv
│   └── test/cic2018_{label}_encoded.csv
├── raw_processed/
│   ├── train/cic2018_{label}_raw_processed.csv
│   └── test/cic2018_{label}_raw_processed.csv
├── pca_cache/
│   ├── cache_pca_train/{label_safe}/pca_*.parquet
│   └── cache_pca_test/{label_safe}/pca_*.parquet
├── embeddings/
│   ├── train/cic2018_{label}_embedding*.parquet
│   └── test/cic2018_{label}_embedding*.parquet
└── encoders/
    ├── pca_major.pkl
    └── umap_major.pkl
```

## Performance Notes

- **Memory Usage**: Peak memory scales with largest label size and chunk/batch sizes
- **Processing Time**: ~2-4 hours for full pipeline on 14M+ rows (depends on hardware)
- **Storage**: PCA cache requires ~2-3x encoded CSV size; embeddings are compact
- **Parallelization**: Per-label processing enables parallel execution across labels

## Troubleshooting

### Common Issues
1. **OOM during UMAP fit**: Reduce class caps or use `--low-memory` flag
2. **NUMBA thread errors**: Set `NUMBA_NUM_THREADS` before import or use `--numba-threads`
3. **Missing PCA cache**: Ensure `pca_transform.py` completed successfully
4. **RowId misalignment**: Verify consistent chunk processing across all steps

### Debug Flags
- Use `--log-level DEBUG` for detailed progress information
- Monitor memory usage with system tools during large operations
- Check intermediate file sizes to ensure proper completion
