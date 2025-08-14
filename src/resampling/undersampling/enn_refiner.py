import pandas as pd
import numpy as np
from imblearn.under_sampling import EditedNearestNeighbours
from tqdm import tqdm
from utils.logging import get_logger

logger = get_logger(__name__)

class ENNRefiner:
    """Apply ENN on a dataframe that contains *at least two* classes.
    Ensures exactly tau_final samples **per class** after refinement.
    If ENN removes too many points it replenishes with random samples
    from the original pool (that were not selected by ENN)."""

    def __init__(self, tau_final: int = 20000, n_neighbors: int = 3, random_state: int = 42):
        self.tau_final = tau_final
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def refine(self, df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
        classes = df[label_col].unique()
        if len(classes) < 2:
            logger.warning("ENNRefiner: need >=2 classes, returning original df")
            return df.copy()

        logger.info(f"[+] Running ENN on {len(df)} samples / {len(classes)} classes …")
        enn = EditedNearestNeighbours(n_neighbors=self.n_neighbors)
        feature_cols = list(df.columns.drop(label_col))
        X_ref, y_ref = enn.fit_resample(
            df[feature_cols],
            df[label_col]
        )
        df_ref = pd.concat([pd.DataFrame(X_ref, columns=feature_cols),
                            pd.Series(y_ref, name=label_col)], axis=1)
        logger.info(f"[+] ENN kept {len(df_ref)} samples")

        # Ensure each class exactly tau_final samples
        balanced_frames = []
        rng = np.random.default_rng(self.random_state)
        for cls in classes:
            subset_enn = df_ref[df_ref[label_col] == cls]
            subset_pool = df[df[label_col] == cls]
            if len(subset_enn) >= self.tau_final:
                balanced_frames.append(subset_enn.sample(n=self.tau_final, random_state=self.random_state))
            else:
                deficit = self.tau_final - len(subset_enn)
                # Anti-join on feature columns to exclude rows already in subset_enn
                tmp = subset_enn[feature_cols].copy()
                tmp['_marker'] = 1
                remaining = subset_pool.merge(tmp, how='left', on=feature_cols, indicator=False)
                remaining = remaining[remaining['_marker'].isna()].drop(columns=['_marker'])
                if len(remaining) < deficit:
                    logger.warning(f"[ENNRefiner] Not enough remaining samples for class {cls}: {len(remaining)} < {deficit}. Sampling with replacement.")
                    add_rows = subset_pool.sample(n=deficit, replace=True, random_state=self.random_state)
                else:
                    add_rows = remaining.sample(n=deficit, replace=False, random_state=self.random_state)
                balanced_frames.append(pd.concat([subset_enn, add_rows], ignore_index=True))
        refined_df = pd.concat(balanced_frames, ignore_index=True)
        logger.info(f"[+] Refined dataset size per class = {self.tau_final}; total {len(refined_df)}")
        return refined_df

