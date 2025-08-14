import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logging import setup_logging, get_logger
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"


def load_before_after():
    before_path = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")
    if not os.path.exists(before_path):
        raise SystemExit(f"Before dataset not found: {before_path}")
    before_df = pd.read_csv(before_path)

    # Require final merged outputs; do not assemble in eval
    final_train = os.path.join(DATA_FOLDER, "cic_final_train_balanced.csv")
    final_test = os.path.join(DATA_FOLDER, "cic_final_test.csv")

    if not (os.path.exists(final_train) and os.path.exists(final_test)):
        raise SystemExit(
            "Final datasets not found. Run 'python -m cli-cic.merge_final' first to create "
            f"{os.path.basename(final_train)} and {os.path.basename(final_test)}."
        )

    after_df = pd.concat([
        pd.read_csv(final_train),
        pd.read_csv(final_test)
    ], ignore_index=True)

    return before_df, after_df


def select_top_classes(df: pd.DataFrame, label_col: str, top_k: int):
    counts = df[label_col].value_counts()
    return counts.index[:top_k].tolist()


def plot_histograms(before_df: pd.DataFrame, after_df: pd.DataFrame, pre: CIC2018Preprocessor,
                    classes: list[str], out_dir: str = "reports/cic_hist",
                    max_features: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    numeric_features = pre.encoded_numerical_features

    for cls in classes:
        before_cls = before_df[before_df[pre.label_column] == cls]
        after_cls = after_df[after_df[pre.label_column] == cls]

        if before_cls.empty or after_cls.empty:
            logger.warning(f"[Histogram] Class {cls} missing in one of the datasets, skip")
            continue

        for feat in numeric_features[:max_features]:
            plt.figure(figsize=(6, 4))
            sns.histplot(before_cls[feat], color="steelblue", stat="density", label="Before", kde=True, alpha=0.6)
            sns.histplot(after_cls[feat], color="orange", stat="density", label="After", kde=True, alpha=0.6)
            plt.title(f"{cls} – {feat}")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(out_dir, f"hist_{cls.replace(' ', '_')}_{feat.replace(' ', '_')}.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            logger.debug(f"[Histogram] Saved {fname}")


if __name__ == "__main__":
    setup_logging("INFO")
    logger.info("[+] Loading CIC datasets …")
    before_df, after_df = load_before_after()

    logger.info("[+] Setting up CIC preprocessor …")
    pre = CIC2018Preprocessor()
    # Ensure only known features are used (drop constants/timestamp)
    before_df = pre.select_features_and_label(before_df)
    after_df = pre.select_features_and_label(after_df)
    # Attempt to load encoders (not strictly required for histogram)
    pre.load_encoders()

    # CLI args
    parser = argparse.ArgumentParser(description="CIC-2018 Histogram evaluation before vs after augmentation")
    parser.add_argument("--top-k-classes", type=int, default=8, help="Top-K most frequent classes to visualize")
    parser.add_argument("--max-features", type=int, default=10, help="Number of numeric features to plot per class")
    args = parser.parse_args()

    classes = select_top_classes(before_df, pre.label_column, args.top_k_classes)
    logger.info(f"[+] Classes selected: {classes}")

    plot_histograms(before_df, after_df, pre, classes, max_features=args.max_features)
    logger.info("[+] CIC Histogram evaluation completed. Check reports/cic_hist/.")


