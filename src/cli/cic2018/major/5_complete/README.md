Finalize & Decode (Major classes)
=================================

## Purpose

Bridge from compressed/coreset selections (in embedding space) back to the original
encoded data and deliver decodable raw_processed CSVs for downstream training/evaluation.

## What it does

For each major label (including Benign/Infilteration):
- Load the standardized compressed selection (train) or selected RowIds (test)
- Map RowIds back to the original encoded CSV (per-label)
- Write an encoded subset: `cic2018_<Label>_encoded_compressed.csv`
- Decode the subset to raw_processed using pre-fitted encoders:
  `cic2018_<Label>_raw_processed_compressed.csv`

## Inputs

- Coreset/test selections (standardized naming):
  - Train: `<base>_compressed_coreset.parquet`
    - Benign uses filtered base: `cic2018_benign_embedding_filtered_compressed_coreset.parquet`
    - Others use base embeddings: `cic2018_<Label>_embedding_compressed_coreset.parquet`
  - Test: `embeddings/test/cic2018_<Label>_test_selected_rowids.parquet`

- Original encoded CSV per label:
  - `encoded/<subset>/cic2018_<Label>_encoded.csv`

## Outputs

- Encoded subset:
  - `encoded/<subset>/cic2018_<Label>_encoded_compressed.csv`
- Raw_processed subset (decoded):
  - `raw_processed/<subset>/cic2018_<Label>_raw_processed_compressed.csv`

## Data flow

RowId propagation guarantees a consistent mapping from embedding selections back to the
original encoded rows. Invalid RowIds (out of range) are logged and ignored.

```
<coreset/test parquet with RowId>
          │
          ▼
 encoded/<subset>/cic2018_<Label>_encoded.csv
          │  (filter by RowId)
          ▼
 encoded/<subset>/cic2018_<Label>_encoded_compressed.csv
          │  (inverse_transform via pre-fitted encoders)
          ▼
 raw_processed/<subset>/cic2018_<Label>_raw_processed_compressed.csv
```

## Usage

Single label:
```bash
python -m cli.cic2018.major.5_complete.finalize_decode \
  --subset train --label "DoS attacks-Hulk" --log-level INFO

python -m cli.cic2018.major.5_complete.finalize_decode \
  --subset test --label "DoS attacks-Hulk" --log-level INFO
```

Batch (all major labels):
```bash
python -m cli.cic2018.major.5_complete.finalize_decode \
  --subset train --label All --log-level INFO

python -m cli.cic2018.major.5_complete.finalize_decode \
  --subset test --label All --log-level INFO
```

## Notes

- Benign train coreset resolves from the filtered embedding base.
- If decoders are not found or decoding fails, the encoded subset is still saved; an error is logged.
- `--numerical-inverse {quantile,minmax}` controls how numerical features are inverse-transformed.


