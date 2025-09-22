Compression (Major classes)
===========================

## Overview

This module provides a fast, simple compression flow for CIC-IDS2018 major classes other than Benign and Infilteration (e.g., DoS attacks-Hulk, Bot, SSH-Bruteforce).

Why this module: these classes do not suffer the same Benign–Infilteration confusion, so we avoid the heavier pipeline (density-aware filtering, boundary detection, 3-component coreset). Instead, we use a simple core + edge strategy to preserve cluster coverage and add diversity.

## Compression strategy

Principle: combine core representatives for coverage with edge points for diversity within each cluster.

1. Micro-clustering: MiniBatchKMeans with K ≈ budget (if dataset >> budget, use K ≈ budget/2)
2. Core selection: pick 1 closest-to-centroid sample per cluster → ensures base coverage
3. Edge selection: within each cluster, take far-from-center points (≥ p95 by `dist_center`) → adds diversity, covers cluster rims
4. Proportional allocation: distribute edge picks across clusters proportionally to their available edge candidates  
5. Top-up and smart trimming: top up with nearest-to-center if still short; if above budget, trim with priority (keep core > edge > top-up)

## Inputs and outputs

Input requirements:
- Embedding parquet must contain `RowId` and `z_*` columns (UMAP transform output)

Columns written in-place (to the base parquet):
- `cluster_id`: MiniBatchKMeans cluster assignment  
- `dist_center`: L2 distance to the assigned cluster center

Standardized outputs:
- Train: `<base>_compressed_coreset.parquet` with `role='core'|'edge'`
- Test: `embeddings/test/cic2018_<Label>_test_selected_rowids.parquet` containing the selected RowIds

Examples
--------
Single label:
```bash
$ python -m cli.cic2018.major.4_others.compression --subset train \
  --label 'DoS attacks-Hulk' --budget 28000 --log-level INFO

2025-09-08 01:25:40 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] rows=402508, dims=24 | budget=28000
2025-09-08 01:25:40 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Using n_clusters=14000
2025-09-08 01:28:22 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(14000, 24) in 162.30 seconds
2025-09-08 01:28:27 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_dd
os_attacks-loic-http_embedding.parquet
2025-09-08 01:28:32 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Final selected: 30834 / 28000 | clusters=14000
2025-09-08 01:28:33 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_ddos_attacks-loic-htt
p_embedding_compressed_coreset.parquet (rows=30834)
2025-09-08 01:28:33 - __main__ - INFO - [DDOS attack-HOIC] rows=139202, dims=24 | budget=28000
2025-09-08 01:28:33 - __main__ - INFO - [DDOS attack-HOIC] Using n_clusters=28000
2025-09-08 01:34:04 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(28000, 24) in 330.87 seconds
2025-09-08 01:34:07 - __main__ - INFO - [DDOS attack-HOIC] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_ddos_att
ack-hoic_embedding.parquet
2025-09-08 01:34:15 - __main__ - INFO - [DDOS attack-HOIC] Final selected: 28000 / 28000 | clusters=28000
2025-09-08 01:34:15 - __main__ - INFO - [DDOS attack-HOIC] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_ddos_attack-hoic_embedding_
compressed_coreset.parquet (rows=28000)
2025-09-08 01:34:16 - __main__ - INFO - [DoS attacks-Hulk] rows=101639, dims=24 | budget=28000
2025-09-08 01:34:16 - __main__ - INFO - [DoS attacks-Hulk] Using n_clusters=28000
2025-09-08 01:39:50 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(28000, 24) in 334.04 seconds
2025-09-08 01:39:52 - __main__ - INFO - [DoS attacks-Hulk] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_dos_atta
cks-hulk_embedding.parquet
2025-09-08 01:39:58 - __main__ - INFO - [DoS attacks-Hulk] Final selected: 28000 / 28000 | clusters=28000
2025-09-08 01:39:58 - __main__ - INFO - [DoS attacks-Hulk] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_dos_attacks-hulk_embedding_
compressed_coreset.parquet (rows=28000)
2025-09-08 01:39:59 - __main__ - INFO - [Bot] rows=101174, dims=24 | budget=28000
2025-09-08 01:39:59 - __main__ - INFO - [Bot] Using n_clusters=28000
2025-09-08 01:45:29 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(28000, 24) in 330.36 seconds
2025-09-08 01:45:31 - __main__ - INFO - [Bot] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_bot_embedding.parquet
2025-09-08 01:45:37 - __main__ - INFO - [Bot] Final selected: 28000 / 28000 | clusters=28000
2025-09-08 01:45:38 - __main__ - INFO - [Bot] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_bot_embedding_compressed_coreset.parquet
 (rows=28000)
2025-09-08 01:45:38 - __main__ - INFO - [SSH-Bruteforce] rows=65833, dims=24 | budget=28000
2025-09-08 01:45:38 - __main__ - INFO - [SSH-Bruteforce] Using n_clusters=28000
2025-09-08 01:51:14 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(28000, 24) in 336.56 seconds
2025-09-08 01:51:16 - __main__ - INFO - [SSH-Bruteforce] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_ssh-brutef
orce_embedding.parquet
2025-09-08 01:51:22 - __main__ - INFO - [SSH-Bruteforce] Final selected: 28000 / 28000 | clusters=28000
2025-09-08 01:51:22 - __main__ - INFO - [SSH-Bruteforce] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_ssh-bruteforce_embedding_comp
ressed_coreset.parquet (rows=28000)
2025-09-08 01:51:22 - __main__ - INFO - [DoS attacks-GoldenEye] rows=28974, dims=24 | budget=28000
2025-09-08 01:51:22 - __main__ - INFO - [DoS attacks-GoldenEye] Using n_clusters=28000
2025-09-08 01:56:44 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(28000, 24) in 322.01 seconds
2025-09-08 01:56:45 - __main__ - INFO - [DoS attacks-GoldenEye] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_dos
_attacks-goldeneye_embedding.parquet
2025-09-08 01:56:49 - __main__ - INFO - [DoS attacks-GoldenEye] Final selected: 28000 / 28000 | clusters=28000
2025-09-08 01:56:49 - __main__ - INFO - [DoS attacks-GoldenEye] Saved compressed set -> /dis/DS/minhtq/CIC-2018//embeddings/train/cic2018_dos_attacks-goldeneye_
embedding_compressed_coreset.parquet (rows=28000)

$ python -m cli.cic2018.major.4_others.compression --subset test --label All --budget 12000 --log-level DEBUG
2025-09-08 08:50:19 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ddos_attacks-loic-http_emb
edding.parquet
2025-09-08 08:50:19 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=172504, z_dims=24 | RowId=True
2025-09-08 08:50:19 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] rows=172504, dims=24 | budget=12000
2025-09-08 08:50:19 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Using n_clusters=6000
2025-09-08 08:50:19 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=6000, batch_size=10000)
2025-09-08 08:51:24 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(6000, 24) in 65.09 seconds
2025-09-08 08:51:25 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ddo
s_attacks-loic-http_embedding.parquet
2025-09-08 08:51:27 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Final selected: 13178 / 12000 | clusters=6000
2025-09-08 08:51:27 - __main__ - INFO - [DDoS attacks-LOIC-HTTP] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_ddos_attacks-loic-http_
test_selected_rowids.parquet (rows=13178)
2025-09-08 08:51:27 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ddos_attack-hoic_embedding
.parquet
2025-09-08 08:51:27 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=59659, z_dims=24 | RowId=True
2025-09-08 08:51:27 - __main__ - INFO - [DDOS attack-HOIC] rows=59659, dims=24 | budget=12000
2025-09-08 08:51:27 - __main__ - INFO - [DDOS attack-HOIC] Using n_clusters=12000
2025-09-08 08:51:27 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=12000, batch_size=10000)
2025-09-08 08:53:39 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12000, 24) in 132.04 seconds
2025-09-08 08:53:39 - __main__ - INFO - [DDOS attack-HOIC] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ddos_atta
ck-hoic_embedding.parquet
2025-09-08 08:53:41 - __main__ - INFO - [DDOS attack-HOIC] Final selected: 12000 / 12000 | clusters=12000
2025-09-08 08:53:41 - __main__ - INFO - [DDOS attack-HOIC] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_ddos_attack-hoic_test_selected_rowids.parquet (rows=12000)
2025-09-08 08:53:41 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_dos_attacks-hulk_embedding.parquet
2025-09-08 08:53:41 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=43560, z_dims=24 | RowId=True
2025-09-08 08:53:41 - __main__ - INFO - [DoS attacks-Hulk] rows=43560, dims=24 | budget=12000
2025-09-08 08:53:41 - __main__ - INFO - [DoS attacks-Hulk] Using n_clusters=12000
2025-09-08 08:53:41 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=12000, batch_size=10000)
2025-09-08 08:55:56 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12000, 24) in 134.63 seconds
2025-09-08 08:55:56 - __main__ - INFO - [DoS attacks-Hulk] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_dos_attacks-hulk_embedding.parquet
2025-09-08 08:55:58 - __main__ - INFO - [DoS attacks-Hulk] Final selected: 12000 / 12000 | clusters=12000
2025-09-08 08:55:58 - __main__ - INFO - [DoS attacks-Hulk] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_dos_attacks-hulk_test_selected_rowids.parquet (rows=12000)
2025-09-08 08:55:58 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_bot_embedding.parquet
2025-09-08 08:55:58 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=43361, z_dims=24 | RowId=True
2025-09-08 08:55:58 - __main__ - INFO - [Bot] rows=43361, dims=24 | budget=12000
2025-09-08 08:55:58 - __main__ - INFO - [Bot] Using n_clusters=12000
2025-09-08 08:55:58 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=12000, batch_size=10000)
2025-09-08 08:58:12 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12000, 24) in 134.74 seconds
2025-09-08 08:58:13 - __main__ - INFO - [Bot] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_bot_embedding.parquet
2025-09-08 08:58:14 - __main__ - INFO - [Bot] Final selected: 12000 / 12000 | clusters=12000
2025-09-08 08:58:14 - __main__ - INFO - [Bot] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_bot_test_selected_rowids.parquet (rows=12000)
2025-09-08 08:58:14 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ssh-bruteforce_embedding.parquet
2025-09-08 08:58:14 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=28215, z_dims=24 | RowId=True
2025-09-08 08:58:14 - __main__ - INFO - [SSH-Bruteforce] rows=28215, dims=24 | budget=12000
2025-09-08 08:58:14 - __main__ - INFO - [SSH-Bruteforce] Using n_clusters=12000
2025-09-08 08:58:14 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=12000, batch_size=10000)
2025-09-08 09:00:27 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12000, 24) in 132.09 seconds
2025-09-08 09:00:27 - __main__ - INFO - [SSH-Bruteforce] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_ssh-bruteforce_embedding.parquet
2025-09-08 09:00:28 - __main__ - INFO - [SSH-Bruteforce] Final selected: 12000 / 12000 | clusters=12000
2025-09-08 09:00:28 - __main__ - INFO - [SSH-Bruteforce] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_ssh-bruteforce_test_selected_rowids.parquet (rows=12000)
2025-09-08 09:00:28 - cli.cic2018.major.helpers.common - DEBUG - Loading embeddings: /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_dos_attacks-goldeneye_embedding.parquet
2025-09-08 09:00:28 - cli.cic2018.major.helpers.common - DEBUG - Loaded embeddings: rows=12418, z_dims=24 | RowId=True
2025-09-08 09:00:28 - __main__ - INFO - [DoS attacks-GoldenEye] rows=12418, dims=24 | budget=12000
2025-09-08 09:00:28 - __main__ - INFO - [DoS attacks-GoldenEye] Using n_clusters=12000
2025-09-08 09:00:28 - resampling.undersampling.kmeans_reps - DEBUG - [KMeansReps] Fitting MiniBatchKMeans(n_clusters=12000, batch_size=10000)
2025-09-08 09:01:29 - resampling.undersampling.kmeans_reps - INFO - [KMeansReps] Fit done. centers shape=(12000, 24) in 60.70 seconds
2025-09-08 09:01:29 - __main__ - INFO - [DoS attacks-GoldenEye] Wrote in-place: cluster_id, dist_center -> /dis/DS/minhtq/CIC-2018//embeddings/test/cic2018_dos_attacks-goldeneye_embedding.parquet
2025-09-08 09:01:30 - __main__ - INFO - [DoS attacks-GoldenEye] Final selected: 12000 / 12000 | clusters=12000
2025-09-08 09:01:30 - __main__ - INFO - [DoS attacks-GoldenEye] Saved compressed set -> /dis/DS/minhtq/CIC-2018/embeddings/test/cic2018_dos_attacks-goldeneye_test_selected_rowids.parquet (rows=12000)
```

Batch (all major labels):
```bash
python -m cli.cic2018.major.4_others.compression --subset train --label All --budget 14000 --log-level INFO

python -m cli.cic2018.major.4_others.compression --subset test \
  --label All --budget 12000 --log-level INFO
```

Tuning knobs
------------
- `--n-clusters`: override K (default ~ budget)
- `--edge-percentile`: 95..99 (default 95)
- `--batch-size`: MiniBatchKMeans batch size (default 10000)
- `--out-path`: explicit output path (optional)

Notes
-----
- If PCA/UMAP embeddings change, re-run to refresh `cluster_id`/`dist_center` and outputs.
- This module is distinct from Benign’s density-aware + boundary + coreset pipeline.


