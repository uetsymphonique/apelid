# CIC-IDS2018 Preprocessing Module

This module handles the initial data preprocessing pipeline for the CIC-IDS2018 dataset, including data cleaning, merging, splitting, and encoder setup.

## Overview

The preprocessing pipeline consists of three main stages:

1. **Data Cleaning & Merging** (`merge_clean.py`)
2. **Train/Test Splitting** (`split_clean_merged.py`) 
3. **Encoder Setup** (`setup_encoders.py`)

## Module Components

### 1. `merge_clean.py`
**Purpose**: Merges raw CIC-IDS2018 CSV files with early data cleaning and splits into per-label files.

**Key Features**:
- Automatic schema detection across multiple CSV files
- Early data type coercion (`float32` for continuous, `Int32` for categorical/binary)
- Sentinel value handling (-1 in `Init Fwd Win Byts`, `Init Bwd Win Byts`)
- Feature selection to reduce memory usage
- Negative value filtering (excluding sentinel columns)
- Per-label file output with deduplication
- Comprehensive logging and diagnostics

**Usage**:
```bash
# Basic run with default settings
python -m cli.cic2018.preprocessing.merge_clean

# Show label distribution per file
python -m cli.cic2018.preprocessing.merge_clean --label-distribution

# Show Protocol value distribution
python -m cli.cic2018.preprocessing.merge_clean --protocol-distribution


```

**Output**: Per-label CSV files in `{DATA_FOLDER}/clean_merged/`
- Format: `cic2018_{label_safe}_clean_merged.csv`
- Example: `cic2018_benign_clean_merged.csv`

### 2. `split_clean_merged.py`
**Purpose**: Splits per-label clean-merged datasets into train/test to prevent data leakage.

**Key Features**:
- Stratified train/test split per label
- Configurable test size (default 30%)
- Label filtering and exclusion options
- Maintains data integrity across splits

**Usage**:
```bash
# Split all labels (default)
python -m cli.cic2018.preprocessing.split_clean_merged

# Split specific labels only
python -m cli.cic2018.preprocessing.split_clean_merged --mode label --labels "Benign" "Infilteration"

# Exclude certain labels
python -m cli.cic2018.preprocessing.split_clean_merged --exclude-labels "Bot" "SSH-Bruteforce"

# Custom test size
python -m cli.cic2018.preprocessing.split_clean_merged --test-size 0.2
```

**Output**: Train/test subdirectories in `{DATA_FOLDER}/clean_merged/`
- `train/cic2018_{label_safe}_train_clean_merged.csv`
- `test/cic2018_{label_safe}_test_clean_merged.csv`

### 3. `setup_encoders.py`
**Purpose**: Fits and saves all necessary encoders (LabelEncoder, OneHotEncoder, MinMaxScaler) using only training data.

**Key Features**:
- Leakage-free encoder fitting (train data only)
- Integrated preprocessing pipeline
- Sentinel value imputation before fitting
- Encoder persistence for consistent transform/inverse operations

**Usage**:
```bash
# Setup encoders with default settings
python -m cli.cic2018.preprocessing.setup_encoders

# Custom sentinel imputation strategy
python -m cli.cic2018.preprocessing.setup_encoders --sentinel-impute median

# Custom input directory
python -m cli.cic2018.preprocessing.setup_encoders --input-dir /path/to/clean_merged
```

**Output**: Encoder files in `{DATA_FOLDER}/encoders/`
- `label_encoder.pkl`
- `onehot_encoder.pkl` 
- `scaler.pkl`

## Complete Pipeline Execution

```bash
# Step 1: Clean and merge raw CSV files
python -m cli.cic2018.preprocessing.merge_clean

# Step 2: Split into train/test
python -m cli.cic2018.preprocessing.split_clean_merged

# Step 3: Setup encoders (train data only)
python -m cli.cic2018.preprocessing.setup_encoders
```

## Data Flow

```
Raw CSV files (10 files, ~14M rows)
    ↓ merge_clean.py
Per-label clean merged files (15 labels)
    ↓ split_clean_merged.py  
Train/test split per label (70/30)
    ↓ setup_encoders.py
Fitted encoders (label, onehot, scaler)
```

## Key Preprocessing Features

- **Schema Harmonization**: Handles inconsistent column sets across CSV files
- **Memory Optimization**: Early feature selection and efficient data types
- **Sentinel Handling**: Special treatment for -1 values in specific columns
- **Data Quality**: Removes infinite, missing, and negative values (with exceptions)
- **Leakage Prevention**: Strict train/test separation before encoder fitting
- **Traceability**: Comprehensive logging for debugging and validation

## Configuration

All paths and settings are configured in `configs.cic2018.py`:
- `ORIGINAL_DATA_FOLDER`: Raw CSV location
- `CLEAN_MERGED_DATA_FOLDER`: Processed data output
- Feature categorization (continuous, binary, categorical)
- Label mappings and safe naming

## Sample Output

```
$ python -m cli.cic2018.preprocessing.merge_clean -d
2025-09-19 15:46:59 - __main__ - INFO - [+] Dropping columns: ['Timestamp', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd PSH Flags', 'Fwd URG Flags', 'FIN Flag Cnt', 'SYN Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count']
2025-09-19 16:05:35 - __main__ - INFO - [+] ===========================================
2025-09-19 16:05:35 - __main__ - INFO - [+] PHASE 1-2 PREPROCESSING COMPLETED!
2025-09-19 16:05:35 - __main__ - INFO - [+] ===========================================
2025-09-19 16:05:35 - __main__ - INFO - [+] Input files processed: 10
2025-09-19 16:05:35 - __main__ - INFO - [+] Per-label files written: 14 @ /dis/DS/minhtq/CIC-2018//clean_merged
2025-09-19 16:05:35 - __main__ - INFO - [+] Final per-label row counts (post-dedup):
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_benign_clean_merged.csv: 6591051
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_ddos_attacks-loic-http_clean_merged.csv: 289289
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_ddos_attack-hoic_clean_merged.csv: 163750
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_bot_clean_merged.csv: 143014
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_ssh-bruteforce_clean_merged.csv: 94043
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_infilteration_clean_merged.csv: 75659
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_dos_attacks-goldeneye_clean_merged.csv: 27772
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_dos_attacks-hulk_clean_merged.csv: 25548
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_dos_attacks-slowloris_clean_merged.csv: 7428
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_brute_force_-web_clean_merged.csv: 270
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_brute_force_-xss_clean_merged.csv: 113
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_sql_injection_clean_merged.csv: 58
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_dos_attacks-slowhttptest_clean_merged.csv: 55
2025-09-19 16:05:35 - __main__ - INFO -     cic2018_ftp-bruteforce_clean_merged.csv: 53
2025-09-19 16:05:35 - __main__ - INFO - [+] ===========================================
<SNIP>

$ python -m cli.cic2018.preprocessing.split_clean_merged --log-level DEBUG
2025-09-19 16:38:04 - __main__ - INFO - ===========================================
2025-09-19 16:38:04 - __main__ - INFO - CLEAN-MERGED PER-LABEL TRAIN/TEST SPLIT STARTED
2025-09-19 16:38:04 - __main__ - INFO - ===========================================
2025-09-19 16:38:04 - __main__ - INFO - Input dir: /dis/DS/minhtq/CIC-2018//clean_merged
2025-09-19 16:38:04 - __main__ - INFO - Files selected: 14
2025-09-19 16:44:22 - __main__ - INFO - benign: total=6591051 -> train=4613735, test=1977316
2025-09-19 16:44:30 - __main__ - INFO - bot: total=143014 -> train=100109, test=42905
2025-09-19 16:44:31 - __main__ - INFO - brute_force_-web: total=270 -> train=189, test=81
2025-09-19 16:44:31 - __main__ - INFO - brute_force_-xss: total=113 -> train=79, test=34
2025-09-19 16:44:40 - __main__ - INFO - ddos_attack-hoic: total=163750 -> train=114625, test=49125
2025-09-19 16:44:57 - __main__ - INFO - ddos_attacks-loic-http: total=289289 -> train=202502, test=86787
2025-09-19 16:44:58 - __main__ - INFO - dos_attacks-goldeneye: total=27772 -> train=19440, test=8332
2025-09-19 16:44:59 - __main__ - INFO - dos_attacks-hulk: total=25548 -> train=17883, test=7665
2025-09-19 16:44:59 - __main__ - INFO - dos_attacks-slowhttptest: total=55 -> train=38, test=17
2025-09-19 16:44:59 - __main__ - INFO - dos_attacks-slowloris: total=7428 -> train=5199, test=2229
2025-09-19 16:45:00 - __main__ - INFO - ftp-bruteforce: total=53 -> train=37, test=16
2025-09-19 16:45:03 - __main__ - INFO - infilteration: total=75659 -> train=52961, test=22698
2025-09-19 16:45:03 - __main__ - INFO - sql_injection: total=58 -> train=40, test=18
2025-09-19 16:45:08 - __main__ - INFO - ssh-bruteforce: total=94043 -> train=65830, test=28213

$ python -m cli.cic2018.preprocessing.setup_encoders --sentinel-impute none
2025-09-19 16:54:09 - __main__ - INFO - ===========================================
2025-09-19 16:54:09 - __main__ - INFO - ENCODER SETUP STARTED
2025-09-19 16:54:09 - __main__ - INFO - ===========================================
2025-09-19 16:54:09 - __main__ - INFO - Scanning TRAIN per-label files in: /dis/DS/minhtq/CIC-2018//clean_merged/train
2025-09-19 16:54:09 - __main__ - INFO - Found 14 TRAIN label files
2025-09-19 16:55:22 - __main__ - INFO - TRAIN union data for encoder fit: (5192667, 65) (from 14 files, 5192667 rows)
2025-09-19 16:55:25 - preprocessing.cic2018_preprocessor - INFO - [+] Unique values for categorical features:
2025-09-19 16:55:25 - preprocessing.cic2018_preprocessor - INFO -     - Protocol: <IntegerArray>
[6]
Length: 1, dtype: Int32
2025-09-19 16:55:40 - preprocessing.cic2018_preprocessor - INFO - [+] Encoders fitted for CIC-2018 dataset
2025-09-19 16:55:40 - preprocessing.cic2018_preprocessor - INFO - [+] Categorical features: ['Protocol']
2025-09-19 16:55:40 - preprocessing.cic2018_preprocessor - INFO - [+] Binary features: ['RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt']
2025-09-19 16:55:40 - preprocessing.cic2018_preprocessor - INFO - [+] Numerical features: 59 features
2025-09-19 16:55:40 - preprocessing.cic2018_preprocessor - INFO - [+] CIC-2018 encoders saved to encoders/cic2018
2025-09-19 16:55:40 - __main__ - INFO - Encoders have been fitted on TRAIN and saved.
2025-09-19 16:55:40 - __main__ - INFO - ===========================================
2025-09-19 16:55:40 - __main__ - INFO - ENCODER SETUP COMPLETED
2025-09-19 16:55:40 - __main__ - INFO - ===========================================
```