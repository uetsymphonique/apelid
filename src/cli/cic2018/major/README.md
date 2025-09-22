CIC-IDS2018 Major Labels Pipeline (Benign, Infilteration, others)
================================================================

This README summarizes the end-to-end processing pipeline for majority classes in CIC-IDS2018, aggregating details from submodules `1_encode`, `2_clustering`, `3_confusion_handle`, `4_others`, and `5_complete`.

## High-level goals

- Produce compact, traceable embeddings with `RowId` for all major labels
- Address Benign ↔ Infilteration confusion with boundary-aware strategies
- Provide fixed-budget compressed train sets and representative test subsets
- Standardize outputs and prefer in-place annotations where appropriate

## Module overview

1. 1_encode — Encode → PCA → UMAP
   - Encode per-label CSVs, fit PCA/UMAP on train, transform train/test to embeddings
   - Uses PCA cache with `RowId` to accelerate UMAP transform; no direct CSV reads
   - Outputs embedding parquet with `z_*` and `RowId`

2. 2_clustering — Clustering and Benign density-aware filtering
   - Benign: density-aware filter on embeddings to ~1.3M with `cluster_id`
   - Others: MiniBatchKMeans clustering; save train centers and optionally write `cluster_id` in-place
   - Artifacts enable fast local kNN and consistent train-test alignment

3. 3_confusion_handle — Boundary, coreset, and test selection (Benign/Infilteration)
   - Boundary detection via relative margin; writes `boundary_score` and `role='boundary'` in-place
   - Train coreset: PIN-first (boundary), then core reps, plus small overlap quota
   - Test selection: multi-component, aligned to train cluster structure

4. 4_others — Simple compression for other major classes
   - Core + edge strategy per cluster under a target budget; writes `cluster_id`, `dist_center`
   - Standardized outputs for train (coreset parquet) and test (selected RowIds)

5. 5_complete — Finalize and decode compressed selections
   - Map selected `RowId` back to original encoded CSVs; save encoded_compressed
   - Decode subsets to raw_processed using pre-fitted encoders

## Data flow (major path)

```
Clean merged → 1_encode (encode → PCA cache → UMAP) → embeddings with RowId
       └─► 2_clustering (Benign filter; others: centers + optional cluster_id)
             └─► 3_confusion_handle (boundary → coreset → test selection)
                   └─► 5_complete (encode_compressed, raw_processed_compressed)
       └─► 4_others (simple compression for non-Benign/Infilteration)
```

## Key strategies

- Relative margin boundary: ratio d_cross / d_same to detect true boundary points
- PIN-first coreset: preserve informative boundary points before core/overlap
- Train-test alignment: test assigned to train centers to preserve structure
- In-place annotations: `cluster_id`, `dist_center`, `boundary_score`, `role`
- RowId propagation: end-to-end traceability from embeddings back to original rows

## Standardized outputs

- Embeddings: `{DATA_FOLDER}/embeddings/{subset}/cic2018_{label}_embedding[_filtered].parquet`
- Centers (train): `{DATA_FOLDER}/embeddings/train/cic2018_{label}_kmeans_centers.npy`
- Benign/Infil train coreset: `..._embedding[_filtered]_compressed_coreset.parquet`
- Other majors (train) coreset: `..._embedding_compressed_coreset.parquet`
- Test selection (RowIds): `embeddings/test/cic2018_{label}_test_selected_rowids.parquet`
- Finalized encoded: `encoded/{subset}/cic2018_{label}_encoded_compressed.csv`
- Finalized raw processed: `raw_processed/{subset}/cic2018_{label}_raw_processed_compressed.csv`

## Typical run order

- Encode and embed:
```bash
python -m cli.cic2018.major.1_encode.encode --subset full
python -m cli.cic2018.major.1_encode.pca_fit
python -m cli.cic2018.major.1_encode.pca_transform --subset train
python -m cli.cic2018.major.1_encode.pca_transform --subset test
python -m cli.cic2018.major.1_encode.umap_fit
python -m cli.cic2018.major.1_encode.umap_transform --subset train
python -m cli.cic2018.major.1_encode.umap_transform --subset test
```

- Clustering & Benign filter:
```bash
python -m cli.cic2018.major.2_clustering.benign_filter
python -m cli.cic2018.major.2_clustering.clustering --label Infilteration --save-cluster-id
# repeat clustering for other major labels as needed
```

- Confusion handling (Benign/Infilteration):
```bash
python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Benign --neighbor Infilteration --use-relative-margin --select
python -m cli.cic2018.major.3_confusion_handle.compute_boundary --target Infilteration --neighbor Benign --use-relative-margin --select
python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Benign --use-relative-margin
python -m cli.cic2018.major.3_confusion_handle.coreset_train --label Infilteration --use-relative-margin
python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Benign
python -m cli.cic2018.major.3_confusion_handle.select_test_ids --label Infilteration
```

- Other majors compression:
```bash
python -m cli.cic2018.major.4_others.compression --subset train --label All --budget 28000
python -m cli.cic2018.major.4_others.compression --subset test --label All --budget 12000
```

- Finalize & decode compressed sets:
```bash
python -m cli.cic2018.major.5_complete.finalize_decode --subset train --label All
python -m cli.cic2018.major.5_complete.finalize_decode --subset test --label All
```

## Notes

- If embeddings or clustering change, re-run downstream steps that depend on them.
- Benign always uses the filtered embedding for boundary/coreset/test.
- Scripts resolve paths via `configs.cic2018.py`; adhere to the standardized file naming.


