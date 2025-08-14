import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.logging import setup_logging, get_logger
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"


def load_datasets():
    before = pd.read_csv(os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv"))

    final_train = os.path.join(DATA_FOLDER, "cic_final_train_balanced.csv")
    final_test = os.path.join(DATA_FOLDER, "cic_final_test.csv")

    if not (os.path.exists(final_train) and os.path.exists(final_test)):
        raise SystemExit(
            "Final datasets not found. Run 'python -m cli-cic.merge_final' first to create "
            f"{os.path.basename(final_train)} and {os.path.basename(final_test)}."
        )

    after = pd.concat([
        pd.read_csv(final_train),
        pd.read_csv(final_test)
    ], ignore_index=True)
    return before, after


def encode_for_tsne(df: pd.DataFrame, pre: CIC2018Preprocessor):
    df_enc = df.copy()
    df_enc = pre.preprocess_encode_binary_features(df_enc)
    df_enc = pre.preprocess_encode_numerical_features(df_enc)
    df_enc = pre.preprocess_encode_categorical_features(df_enc)
    X = df_enc.drop(columns=[pre.label_column]).values.astype(np.float32)
    return X


def sample_per_class(df: pd.DataFrame, label_col: str, n: int):
    frames = []
    for cls, cnt in df[label_col].value_counts().items():
        subset = df[df[label_col] == cls]
        frames.append(subset.sample(n=min(len(subset), n), random_state=1))
    return pd.concat(frames, ignore_index=True)


def run_tsne(df: pd.DataFrame, pre: CIC2018Preprocessor, title: str, out_path: str, max_per_class: int):
    df_sampled = sample_per_class(df, pre.label_column, max_per_class)
    X = encode_for_tsne(df_sampled, pre)
    y = df_sampled[pre.label_column].values

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=40, random_state=1, init="random")
    emb = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 7))
    labels = pd.Series(y).astype(str).unique().tolist()
    cmap = plt.get_cmap('tab20')
    color_map = {lbl: cmap(i % 20) for i, lbl in enumerate(labels)}
    for lbl in labels:
        mask = (y == lbl)
        plt.scatter(emb[mask, 0], emb[mask, 1], s=6, label=str(lbl), alpha=0.6, c=[color_map[lbl]])

    plt.title(title)
    plt.legend(markerscale=2, fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="CIC-2018 t-SNE evaluation before vs after augmentation")
    parser.add_argument("--max-per-class", type=int, default=1500, help="Max samples per class")
    args = parser.parse_args()

    before_df, after_df = load_datasets()
    pre = CIC2018Preprocessor()
    before_df = pre.select_features_and_label(before_df)
    after_df = pre.select_features_and_label(after_df)
    # Load saved encoders if available; otherwise fit on BEFORE
    if not pre.load_encoders():
        pre.setup_encoders(before_df)

    run_tsne(before_df, pre, "CIC-2018 t-SNE BEFORE", "reports/cic_tsne/tsne_before.png", args.max_per_class)
    run_tsne(after_df,  pre, "CIC-2018 t-SNE AFTER",  "reports/cic_tsne/tsne_after.png",  args.max_per_class)
    logger.info("[+] CIC t-SNE evaluation done.")


