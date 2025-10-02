3_minor — Minority per‑label Encoding, Augmentation and Decode
=============================================================

Purpose
-------
Prepare minority-class per-label data across resources (e.g., `cic2018`, `nslkdd`):
- Encode per-label clean_merged CSVs into model feature space
- Augment ENCODED train with WGAN
- Decode augmented ENCODED back to raw_processed for merging

Inputs and Naming
-----------------
- Clean merged per-label (by subset):
  - Train: `{CLEAN_MERGED_DATA_FOLDER}/train/{resources_name}_{label_safe}_train_clean_merged.csv`
  - Test: `{CLEAN_MERGED_DATA_FOLDER}/test/{resources_name}_{label_safe}_test_clean_merged.csv`
- Encoded per-label (by subset):
  - Train: `{ENCODED_DATA_FOLDER}/train/{resources_name}_{label_safe}_encoded.csv`
  - Test: `{ENCODED_DATA_FOLDER}/test/{resources_name}_{label_safe}_encoded.csv`
- Augmented ENCODED (minority, train):
  - `{ENCODED_DATA_FOLDER}/train/{resources_name}_{label_safe}_minority_{strategy}_train_augmented_encoded.csv`
- Decoded augmented RAW (minority, train):
  - `{RAW_PROCESSED_DATA_FOLDER}/train/{resources_name}_{label_safe}_minority_{strategy}_train_augmented_raw_processed.csv`
- `label_safe` = `get_label_name(label)` → lowercase; spaces and `/` → `_`.

Step 1 — Encode (encode.py)
---------------------------
Encodes minority labels into the numeric space expected downstream.

Key ops per label/subset:
- Select features + label from clean_merged
- Numerical encode (required: `--num-encoder {minmax|quantile_uniform}`; must be supported by the resource preprocessor)
- Binary 0/1 encode, label encode, optional categorical one‑hot
- Export ENCODED CSV to `{ENCODED_DATA_FOLDER}/{subset}`

Example:
```bash
python cli/apelid/3_minor/encode.py \
  --resource cic2018 \
  --subset full \
  --mode all \
  --num-encoder quantile_uniform
```

Step 2 — Augment (augmenting.py)
--------------------------------
Generates additional minority ENCODED train samples (currently WGAN).

Workflow:
1) Load ENCODED train per-label
2) Build opposite/benign loader from clean_merged(train) and encode like minority using `--num-encoder`
3) Run WGAN using resource `ACCEPT_RATE_MAP` if available
4) Save augmented ENCODED per‑label train CSV

Required:
- `--resource {cic2018|nslkdd}`
- `--augmenting-strategy wgan`
- `--num-encoder {minmax|quantile_uniform}` (match Step 1)
- `--tau` (target base samples per class)

Example:
```bash
python cli/apelid/3_minor/augmenting.py \
  --resource cic2018 \
  --augmenting-strategy wgan \
  --num-encoder quantile_uniform \
  --tau 65800
```

Step 3 — Decode augmented (decode_augmented.py)
----------------------------------------------
Inverse-transform augmented ENCODED train back to raw_processed.

Example:
```bash
python cli/apelid/3_minor/decode_augmented.py \
  --resource cic2018 \
  --strategy wgan \
  --numerical-inverse quantile_uniform
```

Typical minority pipeline
-------------------------
1) Encode minority per-label (Step 1)
2) Augment minority train per-label (Step 2)
3) Decode augmented to raw_processed (Step 3)
4) Merge with major compressed train (see preparing module)

Troubleshooting
---------------
- “Encoders not found”: ensure encoders for the resource are fitted and loadable
- Unsupported `--num-encoder`: check the resource preprocessor methods exist
- Empty outputs: verify per-label inputs for requested subset/labels
- Use `--log-level DEBUG` for detailed logs


