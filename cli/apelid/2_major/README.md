2_major — Encoding and KMeans-based Downsampling for Training

This module prepares majority-class per-label data for training by:
- encoding raw per-label clean_merged CSVs (feature scaling + label/binary/categorical encoding), and
- downsampling each label with KMeans representatives to a target budget.

It is designed to work with multiple datasets (resources) via a `--resource` flag (e.g., `cic2018`, `nslkdd`).

Inputs and Naming
- Clean merged per-label (by subset):
  - Train: `{CLEAN_MERGED_DATA_FOLDER}/train/{resources_name}_{label_safe}_train_clean_merged.csv`
  - Test: `{CLEAN_MERGED_DATA_FOLDER}/test/{resources_name}_{label_safe}_test_clean_merged.csv`
- Encoded per-label outputs (by subset):
  - Train: `{ENCODED_DATA_FOLDER}/train/{resources_name}_{label_safe}_encoded.csv`
  - Test: `{ENCODED_DATA_FOLDER}/test/{resources_name}_{label_safe}_encoded.csv`
- `label_safe` = `get_label_name(label)` → lowercase, spaces and `/` replaced with `_`.

Step 1 — Encode per-label data (encode.py)
Encode majority labels into a numeric feature space used downstream by models and KMeans.

Key operations (per label and subset):
- Select feature and label columns from clean_merged
- Numerical feature encoding (e.g., Standard/MinMax/Quantile — depends on the resource preprocessor)
- Binary feature normalization (0/1)
- Label encoding (string → int id)
- Optional categorical one-hot encoding
- Export encoded CSV per label to `{ENCODED_DATA_FOLDER}/{subset}`

Essential flags:
```bash
python cli/apelid/2_major/encode.py \
  --resource cic2018 \
  --subset train \
  --mode all             # or: --mode label --labels "Benign" "DoS attacks-Hulk"
```
Notes:
- Encoders must be fitted ahead of time for the selected resource/preprocessor.
- The script validates requested labels against `Resources.MAJORITY_LABELS`.

Step 2 — KMeans compression (kmeans_compression.py)
Select a representative subset per label using KMeans (MiniBatchKMeans by default). This downsampling reduces volume while preserving diversity and cores.

High-level flow for each label:
1) Load encoded train per-label CSV
2) Choose `K` (clusters) from `budget` and strategy
3) Fit KMeans and assign labels
4) Pick core representatives (closest to each cluster centroid)
5) If needed: add “edge” points (top percentile by distance-to-center), then top-up to reach `budget`
6) Map selected encoded rows back to the original per-label clean_merged train via `__rowid__`
7) Write a compressed clean_merged train CSV for the label

Key parameters:
- `--budget` (required): target number of rows to keep per label
- `--strategy`: `core` or `diverse`
  - `core`: only centroid-closest points; optional fallback (`edge_topup` or `random`)
  - `diverse`: core + edges (top-distance percentiles) + top-up; then trim if overshoot
- `--edge-percentile`: edge selection threshold (e.g., 95..99)
- `--n-clusters`: override `K`; default computed as `≈ budget/avg_pts_per_cluster`
- `--avg-pts-per-cluster`: used when `--n-clusters` not set (diverse)
- `--kmeans-algo`: `minibatch` (default) or `full`

Example:
```bash
python cli/apelid/2_major/kmeans_compression.py \
  --resource cic2018 \
  --subset train \
  --mode all \
  --budget 500 \
  --strategy diverse \
  --edge-percentile 95 \
  --kmeans-algo minibatch
```

Outputs:
- Compressed clean_merged train per label:
  - `{CLEAN_MERGED_DATA_FOLDER}/train/{resources_name}_{label_safe}_train_clean_merged_compressed.csv`

Internals — Selection logic
The selection logic is implemented using utilities similar to those in `src/resampling/undersampling/kmeans_reps.py`:
- `compute_default_n_clusters(n_rows, budget, strategy, avg_pts_per_cluster)` chooses `K` when not provided
- `compute_dist_to_center(X, labels, centers)` for edge scoring
- `select_edges_per_cluster(...)` to collect top-percentile edges per cluster
- `distribute_edges(...)` to allocate edges proportionally + round-robin

This makes the workflow predictable and resource-agnostic.

Typical pipeline
1) Fit encoders for the resource (outside this module)
2) Encode majority labels per subset (Step 1)
3) Downsample train per label via KMeans compression (Step 2)
4) Later stages can merge major/minor and prepare final train/test (handled in other modules)

Troubleshooting
- “Encoders not found”: ensure the resource’s preprocessor encoders are fitted and loadable
- Empty outputs: verify clean_merged per-label files exist for the target subset and labels
- Memory: prefer `--kmeans-algo minibatch` for large datasets
- Logs: run with `--log-level DEBUG` to inspect counts for core/edge/top-up/trim

Multi-resource notes
- Use `--resource cic2018` or `--resource nslkdd` to switch datasets
- All file prefixes derive from `resources_name` to keep outputs separate and reproducible


