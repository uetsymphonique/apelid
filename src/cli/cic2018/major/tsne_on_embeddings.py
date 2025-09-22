import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


def _pca_cache_parts(label_safe: str, subset: str) -> list[str]:
    cache_root = os.path.join(cic2018.PCA_CACHE_FOLDER, f"cache_pca_{subset}")
    label_dir = os.path.join(cache_root, label_safe)
    if not os.path.isdir(label_dir):
        raise SystemExit(f"PCA cache directory not found for label {label_safe}: {label_dir}")
    parts = [
        os.path.join(label_dir, f)
        for f in sorted(os.listdir(label_dir))
        if f.endswith('.parquet') and f.startswith('pca_')
    ]
    if not parts:
        raise SystemExit(f"No PCA parts found in {label_dir}")
    return parts


def _load_pca_for_label(label: str,
                        subset: str,
                        cap: int,
                        seed: int,
                        float32: bool,
                        benign_source: str = 'filtered') -> pd.DataFrame:
    """Load PCA rows for a label and sample up to cap.

    - Benign: restrict to RowId in benign filtered embeddings and stratify proportionally by cluster_id from filtered (no re-clustering).
    - Others: simple random sample.
    """
    label_safe = cic2018.get_label_name(label)
    part_files = _pca_cache_parts(label_safe, subset)
    temp = pd.read_parquet(part_files[0])
    pca_cols = [c for c in temp.columns if c.startswith('pca_')]
    if not pca_cols:
        raise SystemExit("No PCA columns in cache part")
    if 'RowId' not in temp.columns:
        raise SystemExit("PCA cache part missing RowId; re-run pca_transform")

    rng = np.random.RandomState(seed)

    # Benign: get allowed RowIds and cluster_id map from embeddings (filtered/base controlled by benign_source)
    allowed_rowids: set[int] | None = None
    cluster_map: dict[int, int] | None = None
    if label == 'Benign':
        use_filtered_benign = (benign_source == 'filtered')
        ben_path = cic2018.embedding_path(subset, 'Benign', filtered_benign=use_filtered_benign)
        if not os.path.exists(ben_path):
            raise SystemExit(f"Benign embeddings not found: {ben_path}")
        # Try to read cluster_id if present; otherwise only RowId
        emb_cols = ['RowId']
        try:
            tmp = pd.read_parquet(ben_path, columns=['RowId', 'cluster_id'])
            has_cluster = True
            emb = tmp
        except Exception:
            emb = pd.read_parquet(ben_path, columns=['RowId'])
            has_cluster = False
        if 'RowId' not in emb.columns:
            raise SystemExit("Benign embeddings must contain RowId")
        allowed_rowids = set(emb['RowId'].astype(np.int64).tolist())
        if has_cluster and 'cluster_id' in emb.columns:
            cluster_map = {int(rid): int(cid) for rid, cid in zip(emb['RowId'].tolist(), emb['cluster_id'].tolist())}
            try:
                n_clusters = int(pd.Series(list(cluster_map.values())).nunique())
            except Exception:
                n_clusters = -1
            logger.debug(f"[TSNE] Benign {benign_source}: rows={len(emb)}, unique clusters={n_clusters}")
        else:
            logger.debug(f"[TSNE] Benign {benign_source}: rows={len(emb)}, cluster_id not present (will sample randomly)")

    # Stream-load PCA with optional restriction and cluster attach
    frames: list[pd.DataFrame] = []
    for pf in part_files:
        df = pd.read_parquet(pf, columns=pca_cols + ['RowId'])
        if float32:
            df[pca_cols] = df[pca_cols].astype(np.float32, copy=False)
        if allowed_rowids is not None:
            df = df[df['RowId'].isin(allowed_rowids)]
            if cluster_map is not None and not df.empty:
                df['cluster_id'] = df['RowId'].map(cluster_map)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all['TSNE_Label'] = label
    logger.info(f"Loaded PCA for {label}: rows={len(df_all)}, cols={len(pca_cols)}")

    # Sampling
    if cap and cap > 0 and len(df_all) > cap:
        if label == 'Benign' and 'cluster_id' in df_all.columns:
            # proportional by cluster_id
            vc = df_all['cluster_id'].value_counts(dropna=False)
            try:
                logger.debug(f"[TSNE] Cluster distribution (top10): {vc.head(10).to_dict()}")
            except Exception:
                pass
            total = int(vc.sum()) if vc.sum() > 0 else 1
            # compute targets
            weights = (vc / float(total)).to_dict()
            # initial floor targets
            targets = {cid: int(np.floor(w * cap)) for cid, w in weights.items()}
            assigned = sum(targets.values())
            # distribute remainder by largest fractional part (approx via weights)
            remainder = int(cap - assigned)
            order = sorted(weights.items(), key=lambda x: -x[1])
            i = 0
            while remainder > 0 and i < len(order):
                cid = order[i][0]
                targets[cid] += 1
                remainder -= 1
                i += 1
            try:
                # Log targets overview
                top_targets = dict(sorted(targets.items(), key=lambda x: -x[1])[:10])
                logger.debug(f"[TSNE] Targets assigned: sum={sum(targets.values())}, remainder_post={remainder}, top10={top_targets}")
            except Exception:
                pass
            # sample per cluster
            chunks: list[pd.DataFrame] = []
            sampled_counts: dict[int, int] = {}
            for cid, need in targets.items():
                if need <= 0:
                    continue
                g = df_all[df_all['cluster_id'] == cid]
                if len(g) == 0:
                    continue
                take = min(need, len(g))
                idx = rng.permutation(len(g))[:take]
                chunks.append(g.iloc[idx])
                sampled_counts[int(cid)] = int(take)
            df_all = pd.concat(chunks, ignore_index=True)
            try:
                logger.debug(f"[TSNE] Sampled per-cluster (top10): {dict(sorted(sampled_counts.items(), key=lambda x: -x[1])[:10])}")
            except Exception:
                pass
            # if slight over-cap due to rounding, trim
            if len(df_all) > cap:
                df_all = df_all.sample(n=cap, random_state=seed).reset_index(drop=True)
                logger.debug(f"[TSNE] Trimmed sampled benign to cap={cap}")
        else:
            df_all = df_all.sample(n=cap, random_state=seed).reset_index(drop=True)
            logger.debug(f"[TSNE] Random sampled {label} to cap={cap}")
    return df_all


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization on PCA space (from PCA cache), flexible component selection")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--components', type=str, nargs='+', default=['Benign', 'Infilteration'], 
                        choices=['Benign', 'Infilteration', 'Boundary_Benign_to_Infil', 'Boundary_Infil_to_Benign'],
                        help='Components to include in t-SNE plot')
    parser.add_argument('--use-coreset', action='store_true', help='Use coreset for Benign/Infilteration instead of full/filtered sets')
    parser.add_argument('--cap-per-label', type=int, default=120000, help='Max rows per label for t-SNE input per class')
    parser.add_argument('--benign-source', type=str, default='base', choices=['base', 'filtered'],
                        help='Choose Benign source for sampling and coreset resolution (base/filtered)')
    parser.add_argument('--exclude-boundary-overlap', action='store_true',
                        help='Exclude boundary points from Benign/Infilteration sets for cleaner visualization')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--n-iter', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=200.0)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--float32', action='store_true')
    parser.add_argument('--save-png', action='store_true', help='Save PNG to REPORT_FOLDER/tsne/<subset>/...')
    parser.add_argument('--save-parquet', action='store_true', help='Save 2D coordinates to DATA_FOLDER/tsne/<subset>/...')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Resolve output fixed paths (PNG -> REPORT_FOLDER, parquet -> DATA_FOLDER)
    comp_names = [c.replace('_', '-') for c in args.components]
    lbl_tag = '_'.join(comp_names)
    mode_suffix = '_coreset' if args.use_coreset else ''
    if args.exclude_boundary_overlap:
        mode_suffix += '_no-boundary-overlap'
    png_dir = os.path.join(cic2018.REPORT_FOLDER, 'tsne', args.subset)
    png_path = os.path.join(png_dir, f"tsne_pca_{args.subset}_{lbl_tag}{mode_suffix}.png")
    parquet_dir = os.path.join(cic2018.DATA_FOLDER, 'tsne', args.subset)
    parquet_path = os.path.join(parquet_dir, f"tsne_pca_{args.subset}_{lbl_tag}{mode_suffix}.parquet")

    frames: list[pd.DataFrame] = []
    
    # Collect boundary RowIds strictly from embeddings (role written in-place)
    benign_boundary_rowids: set[int] = set()
    infil_boundary_rowids: set[int] = set()
    if args.exclude_boundary_overlap:
        # Benign boundary: filtered embeddings with role='boundary'
        ben_path = cic2018.embedding_path(args.subset, 'Benign', filtered_benign=True)
        inf_path = cic2018.embedding_path(args.subset, 'Infilteration', filtered_benign=False)
        if not os.path.exists(ben_path):
            raise SystemExit(f"Benign embeddings not found: {ben_path}")
        if not os.path.exists(inf_path):
            raise SystemExit(f"Infilteration embeddings not found: {inf_path}")
        bdf = pd.read_parquet(ben_path, columns=['RowId', 'role'])
        idf = pd.read_parquet(inf_path, columns=['RowId', 'role'])
        if 'RowId' not in bdf.columns or 'role' not in bdf.columns:
            raise SystemExit("Benign embeddings missing required columns: RowId/role")
        if 'RowId' not in idf.columns or 'role' not in idf.columns:
            raise SystemExit("Infilteration embeddings missing required columns: RowId/role")
        benign_boundary_rowids.update(map(int, bdf.loc[bdf['role'] == 'boundary', 'RowId'].tolist()))
        infil_boundary_rowids.update(map(int, idf.loc[idf['role'] == 'boundary', 'RowId'].tolist()))
        logger.info(f"Collected boundary RowIds from embeddings | Benign={len(benign_boundary_rowids)} Infilteration={len(infil_boundary_rowids)}")

    # Helper to load PCA rows for a RowId selection
    def _load_by_rowids(label: str, rowids: set[int], tag: str, exclude_boundary: bool = False) -> None:
        if exclude_boundary:
            # Only exclude boundary from the same label
            if label == 'Benign' and benign_boundary_rowids:
                original_count = len(rowids)
                rowids = rowids - benign_boundary_rowids
                logger.debug(f"Excluded {original_count - len(rowids)} Benign boundary overlaps from {tag}")
            elif label == 'Infilteration' and infil_boundary_rowids:
                original_count = len(rowids)
                rowids = rowids - infil_boundary_rowids
                logger.debug(f"Excluded {original_count - len(rowids)} Infilteration boundary overlaps from {tag}")
        
        if not rowids:
            logger.warning(f"No RowIds remaining for {tag} after exclusions (skip)")
            return
            
        label_safe = cic2018.get_label_name(label)
        part_files = _pca_cache_parts(label_safe, args.subset)
        temp = pd.read_parquet(part_files[0])
        pca_cols = [c for c in temp.columns if c.startswith('pca_')]
        if not pca_cols or 'RowId' not in temp.columns:
            logger.warning(f"PCA cache for {label} missing required columns (skip)")
            return
        blocks: list[pd.DataFrame] = []
        kept = 0
        for pf in part_files:
            dfp = pd.read_parquet(pf, columns=pca_cols + ['RowId'])
            dfp = dfp[dfp['RowId'].isin(rowids)]
            if dfp.empty:
                continue
            if args.float32:
                dfp[pca_cols] = dfp[pca_cols].astype(np.float32, copy=False)
            blocks.append(dfp)
            kept += len(dfp)
        if kept == 0:
            logger.warning(f"No PCA rows matched provided RowIds for {label} | tag={tag}")
            return
        out = pd.concat(blocks, ignore_index=True)
        out['TSNE_Label'] = tag
        frames.append(out)
        logger.info(f"Added group {tag}: rows={len(out)}")

    subset_dir = os.path.join(cic2018.DATA_FOLDER, 'embeddings', args.subset)
    ben_safe = cic2018.get_label_name('Benign')
    inf_safe = cic2018.get_label_name('Infilteration')

    # Process each requested component
    for comp in args.components:
        if comp == 'Benign':
            if args.use_coreset:
                # Use coreset; resolve from embedding path of chosen source
                ben_embed_path = cic2018.embedding_path(args.subset, 'Benign', filtered_benign=(args.benign_source=='filtered'))
                base, ext = os.path.splitext(ben_embed_path)
                path = f"{base}_compressed_coreset{ext or '.parquet'}"
                if os.path.exists(path):
                    df = pd.read_parquet(path, columns=['RowId'])
                    _load_by_rowids('Benign', set(map(int, df['RowId'].tolist())), 'Benign_Coreset', 
                                  exclude_boundary=args.exclude_boundary_overlap)
                else:
                    logger.warning(f"Benign coreset not found (skip): {path}")
            else:
                # Use chosen Benign source with stratified sampling when possible
                logger.info(f"Loading Benign {args.benign_source} with stratified sampling if cluster_id available")
                dfb = _load_pca_for_label(
                    label='Benign',
                    subset=args.subset,
                    cap=int(args.cap_per_label),
                    seed=int(args.random_state),
                    float32=bool(args.float32),
                    benign_source=str(args.benign_source),
                )
                if args.exclude_boundary_overlap and 'RowId' in dfb.columns and len(benign_boundary_rowids) > 0:
                    before = len(dfb)
                    dfb = dfb[~dfb['RowId'].isin(benign_boundary_rowids)].reset_index(drop=True)
                    logger.debug(f"Excluded {before - len(dfb)} Benign boundary overlaps from non-coreset Benign")
                frames.append(dfb)
                
        elif comp == 'Infilteration':
            if args.use_coreset:
                # Use coreset
                path = os.path.join(subset_dir, f"cic2018_{inf_safe}_embedding_compressed_coreset.parquet")
                if os.path.exists(path):
                    df = pd.read_parquet(path, columns=['RowId'])
                    _load_by_rowids('Infilteration', set(map(int, df['RowId'].tolist())), 'Infilteration_Coreset',
                                  exclude_boundary=args.exclude_boundary_overlap)
                else:
                    logger.warning(f"Infilteration coreset not found (skip): {path}")
            else:
                # Use full embeddings with random sampling
                logger.info(f"Loading Infilteration full with random sampling")
                dfi = _load_pca_for_label(
                    label='Infilteration',
                    subset=args.subset,
                    cap=int(args.cap_per_label),
                    seed=int(args.random_state),
                    float32=bool(args.float32),
                )
                if args.exclude_boundary_overlap and 'RowId' in dfi.columns and len(infil_boundary_rowids) > 0:
                    before = len(dfi)
                    dfi = dfi[~dfi['RowId'].isin(infil_boundary_rowids)].reset_index(drop=True)
                    logger.debug(f"Excluded {before - len(dfi)} Infilteration boundary overlaps from non-coreset Infilteration")
                frames.append(dfi)
                
        elif comp == 'Boundary_Benign_to_Infil':
            # Load strict from embeddings role=boundary (Benign filtered)
            ben_path = cic2018.embedding_path(args.subset, 'Benign', filtered_benign=True)
            if not os.path.exists(ben_path):
                raise SystemExit(f"Benign embeddings not found: {ben_path}")
            bdf = pd.read_parquet(ben_path, columns=['RowId', 'role'])
            if 'RowId' not in bdf.columns or 'role' not in bdf.columns:
                raise SystemExit("Benign embeddings missing required columns: RowId/role")
            rowids = set(map(int, bdf.loc[bdf['role'] == 'boundary', 'RowId'].tolist()))
            if not rowids:
                raise SystemExit("No Benign boundary points found (role='boundary') in embeddings")
            _load_by_rowids('Benign', rowids, 'Boundary_Benign_to_Infil')
                
        elif comp == 'Boundary_Infil_to_Benign':
            inf_path = cic2018.embedding_path(args.subset, 'Infilteration', filtered_benign=False)
            if not os.path.exists(inf_path):
                raise SystemExit(f"Infilteration embeddings not found: {inf_path}")
            idf = pd.read_parquet(inf_path, columns=['RowId', 'role'])
            if 'RowId' not in idf.columns or 'role' not in idf.columns:
                raise SystemExit("Infilteration embeddings missing required columns: RowId/role")
            rowids = set(map(int, idf.loc[idf['role'] == 'boundary', 'RowId'].tolist()))
            if not rowids:
                raise SystemExit("No Infilteration boundary points found (role='boundary') in embeddings")
            _load_by_rowids('Infilteration', rowids, 'Boundary_Infil_to_Benign')
    df_all = pd.concat(frames, ignore_index=True)
    pca_cols = [c for c in df_all.columns if c.startswith('pca_')]
    logger.info(f"Assembled t-SNE input (PCA): rows={len(df_all)}, dims={len(pca_cols)}, labels={dict(df_all['TSNE_Label'].value_counts())}")

    # Run t-SNE
    from sklearn.manifold import TSNE
    X = df_all[pca_cols].to_numpy()
    tsne = TSNE(
        n_components=2,
        perplexity=float(args.perplexity),
        n_iter=int(args.n_iter),
        learning_rate=float(args.learning_rate),
        init='pca',
        random_state=int(args.random_state),
        verbose=0,
        method='barnes_hut',
        angle=0.5,
    )
    Z2 = tsne.fit_transform(X)
    logger.info(f"t-SNE done: shape={Z2.shape}")

    # Optional: save coordinates
    if args.save_parquet:
        out_df = pd.DataFrame({'tsne_x': Z2[:, 0], 'tsne_y': Z2[:, 1], 'TSNE_Label': df_all['TSNE_Label']})
        if 'RowId' in df_all.columns:
            out_df['RowId'] = df_all['RowId'].values
        os.makedirs(parquet_dir, exist_ok=True)
        out_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Saved t-SNE coordinates -> {parquet_path}")

    # Plot PNG
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8), dpi=160)
    labels = df_all['TSNE_Label'].values
    uniq = pd.unique(labels)
    
    # Fixed color mapping for 4 component types
    color_map = {
        'Benign_Coreset': '#1f77b4',           # Blue
        'Infilteration_Coreset': '#ff7f0e',    # Orange  
        'Boundary_Benign_to_Infil': '#2ca02c', # Green
        'Boundary_Infil_to_Benign': '#d62728', # Red
        # Fallback for other labels
        'Benign': '#1f77b4',                   # Blue
        'Infilteration': '#ff7f0e',            # Orange
    }
    
    for u in uniq:
        idx = (labels == u)
        color = color_map.get(u, '#9467bd')  # Default purple for unknown labels
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=2, alpha=0.6, color=color, label=str(u))
    
    plt.legend(markerscale=4, frameon=False)
    plt.title(f"t-SNE on PCA ({args.subset})")
    plt.tight_layout()
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)
        plt.savefig(png_path)
        logger.info(f"Saved t-SNE PNG -> {png_path}")


if __name__ == "__main__":
    main()


