import os
import argparse
import json
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


def _default_inputs() -> tuple[str, str]:
    data_dir = cic2018.DATA_FOLDER
    # Use outputs from preparing/prepare_data_for_train_phase.py (already ordinal-mapped)
    train_path = os.path.join(data_dir, "cic2018_final_train_balanced_cat_map.csv")
    test_path = os.path.join(data_dir, "cic2018_final_test_cat_map.csv")
    return train_path, test_path


def _resolve_cat_features(df: pd.DataFrame, label_col: str, user_cols: list[str] | None, max_cardinality: int) -> list[str]:
    if user_cols:
        missing = [c for c in user_cols if c not in df.columns]
        if missing:
            logger.warning(f"Categorical columns not found and will be ignored: {missing}")
        return [c for c in user_cols if c in df.columns]

    # Heuristic: integer/low-cardinality columns (excluding label) as categorical
    candidates: list[str] = []
    for col in df.columns:
        if col == label_col:
            continue
        if col.startswith("__"):
            continue
        s = df[col]
        # Integer-like dtype
        if pd.api.types.is_integer_dtype(s) or (pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all()):
            try:
                nunq = int(s.nunique(dropna=True))
                if nunq <= max_cardinality:
                    candidates.append(col)
            except Exception:
                pass
    return candidates


def main():
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as e:
        raise SystemExit(f"CatBoost is required. Please install: pip install catboost. Error: {e}")

    parser = argparse.ArgumentParser(description="Train CatBoost on merged CIC2018 raw_processed train/test (no extra encoding)")
    parser.add_argument('--train-in', type=str, default=_default_inputs()[0])
    parser.add_argument('--test-in', type=str, default=_default_inputs()[1])
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--drop-cols', type=str, nargs='*', default=['__source__'], help='Extra columns to drop from features')
    parser.add_argument('--cat-cols', type=str, nargs='*', default=None, help='Categorical column names (override auto)')
    parser.add_argument('--cat-max-card', type=int, default=256, help='Auto-detect categorical if unique values <= this')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction from train for early stopping')
    parser.add_argument('--eval-metric', type=str, default='TotalF1', choices=['TotalF1', 'WeightedF1'], help='Evaluation metric')
    parser.add_argument('--iterations', type=int, default=4000)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.08)
    parser.add_argument('--l2-leaf-reg', type=float, default=3.0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--auto-class-weights', type=str, default='Balanced', choices=['None', 'Balanced', 'SqrtBalanced'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'CPU', 'GPU'], help='Train on CPU/GPU (auto tries GPU then CPU)')
    parser.add_argument('--gpu-devices', type=str, default='0', help='Comma-separated GPU device indices (when --device GPU/auto)')
    parser.add_argument('--model-out', type=str, default=os.path.join(cic2018.DATA_FOLDER, 'models', 'cic2018_catboost.cbm'))
    parser.add_argument('--report-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'metrics.json'))
    parser.add_argument('--cm-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'confusion_matrix.png'))
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load data
    if not os.path.exists(args.train_in) or not os.path.exists(args.test_in):
        raise SystemExit(f"Input files not found: train={args.train_in}, test={args.test_in}")
    logger.info(f"[+] Loading train/test: {args.train_in} | {args.test_in}")
    train_df = pd.read_csv(args.train_in, low_memory=False)
    test_df = pd.read_csv(args.test_in, low_memory=False)
    logger.info(f"[+] Shapes: train={train_df.shape}, test={test_df.shape}")

    # Prepare features/labels
    label_col = args.label_col
    if label_col not in train_df.columns:
        raise SystemExit(f"Label column '{label_col}' not found in train")
    if label_col not in test_df.columns:
        raise SystemExit(f"Label column '{label_col}' not found in test")

    drop_cols = [c for c in (args.drop_cols or []) if c in train_df.columns]
    feat_cols = [c for c in train_df.columns if c not in drop_cols and c != label_col]
    # Align columns across train/test
    common_cols = [c for c in feat_cols if c in test_df.columns]
    if len(common_cols) != len(feat_cols):
        logger.warning("Feature columns differ between train and test; aligning to intersection")
    X_train_all = train_df[common_cols].copy()
    y_train_all = train_df[label_col].copy()
    X_test = test_df[common_cols].copy()
    y_test = test_df[label_col].copy()

    # Cat features resolution
    cat_cols = _resolve_cat_features(train_df[common_cols + [label_col]], label_col, args.cat_cols, args.cat_max_card)
    cat_idx = [X_train_all.columns.get_loc(c) for c in cat_cols if c in X_train_all.columns]
    logger.info(f"[+] Categorical features: {cat_cols} (idx={cat_idx})")

    # Split validation from train
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_all, test_size=args.val_frac, random_state=args.random_seed, stratify=y_train_all if y_train_all.nunique() > 1 else None
    )

    # X_tr = X_train_all.copy()
    # y_tr = y_train_all.copy()
    # X_val = X_test.copy()
    # y_val = y_test.copy()

    # Cast categorical columns to integer (prepare outputs are ordinal-mapped; avoid float like 6.0)
    def _cast_cats_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return df
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce').fillna(-1).astype(np.int32)
        return out

    X_tr = _cast_cats_to_int(X_tr, cat_cols)
    X_val = _cast_cats_to_int(X_val, cat_cols)
    X_test = _cast_cats_to_int(X_test, cat_cols)

    # Pools
    train_pool = Pool(data=X_tr, label=y_tr, cat_features=cat_idx if cat_idx else None)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_idx if cat_idx else None)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_idx if cat_idx else None)

    # Model
    auto_class_weights = None if args.auto_class_weights == 'None' else args.auto_class_weights
    def _build_model(task_type: str):
        params = dict(
            iterations=int(args.iterations),
            depth=int(args.depth),
            learning_rate=float(args.learning_rate),
            l2_leaf_reg=float(args.l2_leaf_reg),
            random_seed=int(args.random_seed),
            # early_stopping_rounds=20,
            loss_function='MultiClass',
            eval_metric=args.eval_metric,
            # bootstrap_type='Bernoulli',
            od_type='Iter',
            od_wait=100,
            verbose=100,
            auto_class_weights=auto_class_weights,
            task_type=task_type,
        )
        if task_type == 'GPU':
            params['devices'] = args.gpu_devices
        return CatBoostClassifier(**params)

    # Choose device
    model = None
    last_err = None
    if args.device in ('GPU', 'auto'):
        try:
            logger.info("[+] Trying CatBoost on GPU …")
            model = _build_model('GPU')
        except Exception as e:
            last_err = e
            logger.warning(f"[!] Failed to initialize GPU CatBoost, will try CPU. Err: {e}")
    if model is None:
        logger.info("[+] Using CatBoost on CPU …")
        model = _build_model('CPU')

    # Fit
    logger.info("[+] Training CatBoost …")
    try:
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    except Exception as e:
        if getattr(model, 'get_param', lambda k: None)('task_type') == 'GPU' and args.device == 'auto':
            logger.warning(f"[!] GPU training failed, falling back to CPU. Err: {e}")
            model = _build_model('CPU')
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        else:
            raise

    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
    logger.info("[+] Evaluating on test …")
    y_pred = model.predict(test_pool)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average='weighted'))
    prem = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recm = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save model and metrics
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save_model(args.model_out)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump({
            'accuracy': acc,
            'f1_weighted': f1m,
            'precision_weighted': prem,
            'recall_weighted': recm,
            'classification_report': report,
            'confusion_matrix': cm,
            'features': list(X_train_all.columns),
            'cat_features': cat_cols,
        }, f, indent=2)

    logger.info(f"[+] Saved model => {args.model_out}")
    logger.info(f"[+] Test accuracy={acc:.4f}, weighted-F1={f1m:.4f}, weighted-Precision={prem:.4f}, weighted-Recall={recm:.4f}")
    logger.info(f"[+] Metrics JSON => {args.report_out}")

    # Log and plot confusion matrix
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        labels_sorted = sorted(np.unique(np.concatenate([y_test.values, np.array(y_pred)])))
        cm_mat = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        # Log CM to stdout
        logger.debug("Confusion matrix (rows=true, cols=pred):")
        header = "labels," + ",".join(map(str, labels_sorted))
        logger.debug(header)
        for i, row in enumerate(cm_mat):
            logger.debug(f"{labels_sorted[i]}," + ",".join(str(int(x)) for x in row))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_mat, display_labels=labels_sorted)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
        os.makedirs(os.path.dirname(args.cm_out), exist_ok=True)
        plt.tight_layout()
        fig.savefig(args.cm_out, dpi=150)
        plt.close(fig)
        logger.info(f"[+] Confusion matrix PNG => {args.cm_out}")
    except Exception as e:
        logger.warning(f"[!] Failed to plot confusion matrix: {e}")


if __name__ == "__main__":
    main()


