Minor Classes Pipeline (Encode → Augment → Decode)
=================================================

## Purpose

Provide a lightweight pipeline for CIC-IDS2018 minority classes: encode per-label
datasets, augment encoded train with WGAN (optionally others), and decode back to
raw_processed for downstream merging and training.

## Components

1) encode.py — per-label encoding (no PCA/UMAP)
- One-hot/ordinal for categorical, binary encoders
- MinMax or Quantile (uniform) encoding for numerical features
- Optional handling of -1 sentinel values with indicator columns
- Outputs per-label encoded CSVs under `encoded/{subset}/`

2) augmenting.py — augmentation on encoded train
- WGAN support (CFM/FDM placeholders)
- Uses Benign major compressed encoded train as critic reference for WGAN
- Ensures a valid `test_df` for WGAN dedup (loads encoded test or samples train)
- Saves augmented encoded train per label

3) decode_augmented.py — inverse-transform augmented encoded
- Decodes augmented encoded outputs back to raw_processed format
- Uses the same pre-fitted encoders as the major pipeline

## Data flow

```
clean_merged/<subset>/cic2018_<label>_*_clean_merged.csv
   ↓ encode.py (per-label)
encoded/<subset>/cic2018_<label>_encoded.csv
   ↓ augmenting.py (train only)
encoded/train/cic2018_<label>_minority_wgan_train_augmented_encoded.csv
   ↓ decode_augmented.py
raw_processed/train/cic2018_<label>_minority_wgan_train_augmented_raw_processed.csv
```

## Inputs/Outputs

Inputs:
- Clean merged per-label CSVs: `clean_merged/{subset}/cic2018_<label>_*_clean_merged.csv`
- Pre-fitted encoders (same as major pipeline)
- For WGAN: optional encoded test per label for dedup

Outputs:
- Encoded per label: `encoded/{subset}/cic2018_<label>_encoded.csv`
- Augmented encoded train: `encoded/train/cic2018_<label>_minority_wgan_train_augmented_encoded.csv`
- Decoded augmented train: `raw_processed/train/cic2018_<label>_minority_wgan_train_augmented_raw_processed.csv`

## Usage

Encode:
```bash
python -m cli.cic2018.minor.encode --subset train --log-level INFO
python -m cli.cic2018.minor.encode --subset test --log-level INFO
# Or full mode to process both when available
python -m cli.cic2018.minor.encode --subset full --log-level INFO
```

Augment (WGAN):
```bash
python -m cli.cic2018.minor.augmenting --augmenting-strategy wgan --mode all --tau 28000 --log-level INFO
# Single label
python -m cli.cic2018.minor.augmenting --augmenting-strategy wgan --mode label --label "FTP-BruteForce" --tau 28000
```

Decode augmented:
```bash
python -m cli.cic2018.minor.decode_augmented --strategy wgan --mode all --log-level INFO
# Single label to a custom dir
python -m cli.cic2018.minor.decode_augmented --strategy wgan --mode label --labels "FTP-BruteForce" --out-dir /tmp/decoded
```

## Notes

- Always fit and load encoders from TRAIN to avoid leakage.
- WGAN expects a valid test_df for dedup; if missing, the script samples from train.
- File names are standardized; use `configs.cic2018` for path resolution and label names.


