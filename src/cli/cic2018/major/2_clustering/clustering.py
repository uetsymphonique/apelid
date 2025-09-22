import os
import argparse
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from ..helpers import load_embeddings
from resampling.undersampling.kmeans_reps import KMeansRepresentativeSelector


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings for a label and save MiniBatchKMeans centers; optionally save cluster_id per point")
    parser.add_argument('--subset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--label', type=str, choices=['Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
        'Infilteration','SSH-Bruteforce', 'DoS attacks-GoldenEye'])
    parser.add_argument('--input-path', type=str, default=None,
                        help='Parquet embeddings path (default resolves from embeddings/<subset>/)')
    parser.add_argument('--benign-source', type=str, default='base', choices=['filtered', 'base'],
                        help='For Benign: choose embeddings source and encode into centers filename')
    parser.add_argument('--k', type=int, default=1500, help='Number of clusters (default: 1500)')
    parser.add_argument('--batch-size', type=int, default=10000, help='MiniBatchKMeans batch size (default: 10000)')
    parser.add_argument('--save-centers', type=str, default=None,
                        help='Path to save centers as .npy/.npz (default embeddings/train/cic2018_<label>_kmeans_centers.npy for train)')
    parser.add_argument('--save-cluster-id', action='store_true', help='Write cluster_id column back into the input parquet (in-place)')
    parser.add_argument('--float32', action='store_true', default=True)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    label_safe = cic2018.get_label_name(args.label)
    if args.input_path is None:
        filtered_benign = (args.label == 'Benign' and args.benign_source == 'filtered')
        args.input_path = cic2018.embedding_path(args.subset, args.label, filtered_benign=filtered_benign)
    if not os.path.exists(args.input_path):
        raise SystemExit(f"Embedding parquet not found: {args.input_path}")

    df = load_embeddings(args.input_path, float32=args.float32)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    X = df[z_cols].to_numpy(copy=False)
    logger.info(f"Loaded embeddings: rows={len(df)}, dims={len(z_cols)}")

    # Fit KMeans via helper class
    selector = KMeansRepresentativeSelector(n_clusters=int(args.k), batch_size=int(args.batch_size), random_state=42)
    selector.fit(X)
    centers = selector.centers_
    if centers is None:
        raise SystemExit("KMeans centers not available after fit")
    logger.info(f"Fitted centers: shape={centers.shape}")

    # Resolve centers path
    if args.save_centers is None:
        benign_source = args.benign_source if args.label == 'Benign' else None
        args.save_centers = cic2018.kmeans_centers_path_train(args.label, benign_source=benign_source)
    os.makedirs(os.path.dirname(args.save_centers), exist_ok=True)
    np.save(args.save_centers, centers)
    logger.info(f"Saved centers -> {args.save_centers}")

    # Optionally write cluster_id in-place
    if args.save_cluster_id:
        labels = selector.predict(X)
        df_out = pd.read_parquet(args.input_path)
        df_out['cluster_id'] = labels.astype(np.int32, copy=False)
        df_out.to_parquet(args.input_path, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Wrote cluster_id in-place -> {args.input_path}")


if __name__ == "__main__":
    main()
