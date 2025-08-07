import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import setup_logging, get_logger
import argparse

# -------------------------------------------------------------
# t-SNE Evaluation Script
# -------------------------------------------------------------
#  • Load BEFORE (majority + minority raw) and AFTER (final_train + final_test)
#  • Encode features numerically (ordinal for categorical, min-max for numeric)
#  • Sample max N per class to keep plot readable
#  • Plot 5 classes on the same 2-D t-SNE scatter
#    – before_tsne.png , after_tsne.png in reports/tsne/
# -------------------------------------------------------------

CLASSES = ["Benign", "DoS", "Probe", "R2L", "U2R"]
SAMPLE_PER_CLASS = 14000  # adjust if memory issues

logger = get_logger(__name__)


def load_datasets(base_dir="data"):
    before = pd.concat([
        pd.read_csv(os.path.join(base_dir, "majority_classes_raw.csv")),
        pd.read_csv(os.path.join(base_dir, "minority_classes_raw.csv"))
    ], ignore_index=True)

    after = pd.concat([
        pd.read_csv(os.path.join(base_dir, "final_train_balanced.csv")),
        pd.read_csv(os.path.join(base_dir, "final_test.csv"))
    ], ignore_index=True)
    return before, after


def encode_for_tsne(df: pd.DataFrame, pre: NSLKDDPreprocessor):
    # Ordinal encode cat → int, scale numeric 0-1, keep binary 0/1
    df_enc = df.copy()
    df_enc = pre.preprocess_encode_binary_features(df_enc)
    df_enc = pre.preprocess_encode_numerical_features(df_enc)
    df_enc = pre.preprocess_encode_categorical_features(df_enc)
    X = df_enc.drop(columns=[pre.label_column]).values.astype(np.float32)
    return X


def sample_per_class(df: pd.DataFrame, n: int):
    frames = []
    for cls in CLASSES:
        subset = df[df["Label"] == cls]
        if subset.empty:
            logger.warning(f"Class {cls} missing – skipping")
            continue
        if n < 0:
            frames.append(subset)
        else:
            frames.append(subset.sample(n=min(len(subset), n), random_state=1))
    return pd.concat(frames, ignore_index=True)


def run_tsne(df: pd.DataFrame, pre: NSLKDDPreprocessor, title: str, out_path: str, max_per_class: int):
    df_sampled = sample_per_class(df, max_per_class)
    X = encode_for_tsne(df_sampled, pre)
    y = df_sampled[pre.label_column].values

    # Standardize before TSNE for better geometry
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, random_state=1, init="random")
    emb = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(7,6))
    colors = {
        "Benign": "#1f77b4",
        "DoS": "#ff7f0e",
        "Probe": "#2ca02c",
        "R2L": "#d62728",
        "U2R": "#9467bd"
    }
    for cls in CLASSES:
        mask = y == cls
        plt.scatter(emb[mask,0], emb[mask,1], s=6, label=cls, alpha=0.6, c=colors.get(cls, None))

    plt.title(title)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    setup_logging("INFO")
    logger.info("[+] Loading datasets …")
    before_df, after_df = load_datasets()

    logger.info("[+] Setting up preprocessor (onehot for categorical, min-max for numeric) …")
    pre = NSLKDDPreprocessor()
    # Need to fit encoders on BEFORE to get ordinal mapping & scaler
    pre.setup_encoders(before_df)

    # parse CLI args
    parser = argparse.ArgumentParser(description="t-SNE evaluation before vs after augmentation")
    parser.add_argument("--max-per-class", type=int, default=1500,
                        help="Max samples per class (-1 for all)")
    args = parser.parse_args()

    run_tsne(before_df, pre, "t-SNE BEFORE augmentation", "reports/tsne/tsne_before.png", args.max_per_class)
    run_tsne(after_df,  pre, "t-SNE AFTER augmentation",  "reports/tsne/tsne_after.png", args.max_per_class)
    logger.info("[+] t-SNE evaluation done.") 