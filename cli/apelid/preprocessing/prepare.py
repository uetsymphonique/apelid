import numpy as np
import pandas as pd
from .preprocessor import Preprocessor
from sklearn.model_selection import train_test_split


class PrepareData:
    @staticmethod
    def prepare_training_data(
        df: pd.DataFrame,
        pre: Preprocessor,
        *,
        encode_numerical: bool = False,
        use_validation: bool = False,
        val_size: float = 0.1,
        random_state: int = 42,
        mode: str = 'tree',  # 'tree' | 'dnn'
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
        """Return numpy arrays and metadata for training.

        Returns: X_tr, X_val, y_tr, y_val, meta
        meta: { feature_names, class_names, cat_feature_indices, cont_feature_indices }
        """
        train_df = pre.select_features_and_label(df.copy())
        # Ordinal for categorical
        train_df = pre.preprocess_encode_ordinal_features(train_df)
        # Binary to 0/1
        train_df = pre.preprocess_encode_binary_features(train_df)
        # Numerical scaling if DNN mode requested
        if encode_numerical or mode == 'dnn':
            train_df = pre.preprocess_encode_numerical_features_standard(train_df)
        # Label encode
        train_df = pre.preprocess_encode_label(train_df)

        # Split
        if use_validation:
            strat = train_df[pre.label_column] if train_df[pre.label_column].nunique() > 1 else None
            train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state, stratify=strat)
        else:
            val_df = None

        feature_names = list(pre.ordered_features)
        X_tr = train_df[feature_names].to_numpy()
        y_tr = train_df[pre.label_column].to_numpy()
        X_val = val_df[feature_names].to_numpy() if val_df is not None else None
        y_val = val_df[pre.label_column].to_numpy() if val_df is not None else None

        # Build indices for cat/cont if available on preprocessor
        cat_cols = getattr(pre, 'encoded_categorical_features_ordinal', []) or []
        cont_cols = getattr(pre, 'cont_features', []) or []
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        cat_idx = [name_to_idx[c] for c in cat_cols if c in name_to_idx]
        cont_idx = [name_to_idx[c] for c in cont_cols if c in name_to_idx]
        class_names = list(getattr(getattr(pre, 'encoders', {}), 'get', lambda k: None)('label').classes_) if getattr(pre, 'encoders', None) and 'label' in pre.encoders else []

        meta = {
            'feature_names': feature_names,
            'class_names': class_names,
            'cat_feature_indices': cat_idx,
            'cont_feature_indices': cont_idx,
        }
        return X_tr, X_val, y_tr, y_val, meta

    @staticmethod
    def prepare_input_data(
        df: pd.DataFrame,
        pre: Preprocessor,
        *,
        encode_numerical: bool = False,
        include_label: bool = True,
        mode: str = 'tree',
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Return numpy arrays and metadata for test/inference.

        Returns: X_te, y_te (or None), meta
        """
        input_df = pre.select_features_and_label(df.copy()) if include_label else df.copy()
        input_df = pre.preprocess_encode_ordinal_features(input_df)
        input_df = pre.preprocess_encode_binary_features(input_df)
        if encode_numerical or mode == 'dnn':
            input_df = pre.preprocess_encode_numerical_features_standard(input_df)
        if include_label:
            input_df = pre.preprocess_encode_label(input_df)

        feature_names = list(pre.ordered_features)
        X_te = input_df[feature_names].to_numpy()
        y_te = input_df[pre.label_column].to_numpy() if include_label and pre.label_column in input_df.columns else None

        cat_cols = getattr(pre, 'encoded_categorical_features_ordinal', []) or []
        cont_cols = getattr(pre, 'cont_features', []) or []
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        cat_idx = [name_to_idx[c] for c in cat_cols if c in name_to_idx]
        cont_idx = [name_to_idx[c] for c in cont_cols if c in name_to_idx]
        class_names = list(getattr(getattr(pre, 'encoders', {}), 'get', lambda k: None)('label').classes_) if getattr(pre, 'encoders', None) and 'label' in pre.encoders else []

        meta = {
            'feature_names': feature_names,
            'class_names': class_names,
            'cat_feature_indices': cat_idx,
            'cont_feature_indices': cont_idx,
        }
        return X_te, y_te, meta