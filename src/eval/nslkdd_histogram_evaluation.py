import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import setup_logging, get_logger

# -----------------------------------------------------------------------------
# Histogram Evaluation Script
# -----------------------------------------------------------------------------
# 1. Load raw BEFORE dataset  (majority + minority raw)
# 2. Load AFTER dataset (final_train_balanced + final_test)
# 3. Với mỗi trong 5 lớp [Benign, DoS, Probe, R2L, U2R] và mỗi numeric feature
#    vẽ histogram overlay BEFORE vs AFTER (density) và lưu vào reports/.
# -----------------------------------------------------------------------------

def load_before_after(base_dir="data"):
    before_majority = pd.read_csv(os.path.join(base_dir, "majority_classes_raw.csv"))
    before_minority = pd.read_csv(os.path.join(base_dir, "minority_classes_raw.csv"))
    before_df = pd.concat([before_majority, before_minority], ignore_index=True)

    after_train = pd.read_csv(os.path.join(base_dir, "final_train_balanced.csv"))
    after_test  = pd.read_csv(os.path.join(base_dir, "final_test.csv"))
    after_df = pd.concat([after_train, after_test], ignore_index=True)
    return before_df, after_df


def plot_histograms(before_df: pd.DataFrame, after_df: pd.DataFrame, pre: NSLKDDPreprocessor,
                     out_dir="reports/nslkdd_hist"):
    os.makedirs(out_dir, exist_ok=True)

    numeric_features = pre.encoded_numerical_features
    classes = ["Benign", "DoS", "Probe", "R2L", "U2R"]

    for cls in classes:
        before_cls = before_df[before_df[pre.label_column] == cls]
        after_cls  = after_df[after_df[pre.label_column] == cls]

        if before_cls.empty or after_cls.empty:
            logger.warning(f"Class {cls} missing in one of the datasets, skip")
            continue

        for feat in numeric_features[:10]:  # limit to first 10 features to save time
            plt.figure(figsize=(6,4))
            sns.histplot(before_cls[feat], color="steelblue", stat="density", label="Before", kde=True, alpha=0.6)
            sns.histplot(after_cls[feat],  color="orange",    stat="density", label="After" , kde=True, alpha=0.6)
            plt.title(f"{cls} – {feat}")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(out_dir, f"hist_{cls}_{feat}.png")
            plt.savefig(fname, dpi=300)
            plt.close()
            logger.debug(f"Saved {fname}")


if __name__ == "__main__":
    setup_logging("INFO")
    logger = get_logger(__name__)

    logger.info("[+] Loading datasets …")
    before_df, after_df = load_before_after()

    logger.info("[+] Initialising preprocessor …")
    pre = NSLKDDPreprocessor()
    # No need to load encoders for raw histogram; we just use feature lists

    logger.info("[+] Plotting histograms …")
    plot_histograms(before_df, after_df, pre)
    logger.info("[+] Histogram evaluation completed. Check reports/hist/.") 