import os
import argparse
import json
import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


def _default_inputs() -> tuple[str, str]:
    data_dir = cic2018.DATA_FOLDER
    train_path = os.path.join(data_dir, "cic2018_final_train_balanced_cat_map.csv")
    test_path = os.path.join(data_dir, "cic2018_final_test_cat_map.csv")
    return train_path, test_path


def _resolve_cat_features(df: pd.DataFrame, label_col: str, user_cols: list[str] | None, max_cardinality: int) -> list[str]:
    if user_cols:
        missing = [c for c in user_cols if c not in df.columns]
        if missing:
            logger.warning(f"Categorical columns not found and will be ignored: {missing}")
        return [c for c in user_cols if c in df.columns]
    # Heuristic: mapped integer flags as categorical
    candidates: list[str] = ['RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt']
    return [c for c in candidates if c in df.columns]


def main():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception as e:
        raise SystemExit(f"scikit-learn is required. Please install: pip install scikit-learn. Error: {e}")

    parser = argparse.ArgumentParser(description="Train GradientBoostingClassifier (GBM) or HistGradientBoosting on CIC2018")
    parser.add_argument('--train-in', type=str, default=_default_inputs()[0])
    parser.add_argument('--test-in', type=str, default=_default_inputs()[1])
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--drop-cols', type=str, nargs='*', default=['__source__'])
    parser.add_argument('--cat-cols', type=str, nargs='*', default=None)
    parser.add_argument('--cat-max-card', type=int, default=256)
    parser.add_argument('--val-frac', type=float, default=0.1)
    # Implementation choice
    parser.add_argument('--impl', type=str, default='gbdt', choices=['gbdt', 'hist'])
    # GBM params
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--subsample', type=float, default=0.9)
    # HistGradientBoosting options
    parser.add_argument('--early-stopping', type=int, default=30, help='Only for --impl hist: n_iter_no_change')
    parser.add_argument('--n-threads', type=int, default=0, help='For --impl hist: set OMP_NUM_THREADS if >0')
    parser.add_argument('--random-seed', type=int, default=42)
    # Outputs
    parser.add_argument('--model-out', type=str, default=os.path.join(cic2018.DATA_FOLDER, 'models', 'cic2018_gbm.joblib'))
    parser.add_argument('--report-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'gbm', 'metrics.json'))
    parser.add_argument('--cm-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'gbm', 'confusion_matrix.png'))
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not os.path.exists(args.train_in) or not os.path.exists(args.test_in):
        raise SystemExit(f"Input files not found: train={args.train_in}, test={args.test_in}")
    logger.info(f"[+] Loading train/test: {args.train_in} | {args.test_in}")
    train_df = pd.read_csv(args.train_in, low_memory=False)
    test_df = pd.read_csv(args.test_in, low_memory=False)
    logger.info(f"[+] Shapes: train={train_df.shape}, test={test_df.shape}")

    label_col = args.label_col
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise SystemExit("Label column not found in inputs")

    drop_cols = [c for c in (args.drop_cols or []) if c in train_df.columns]
    feat_cols = [c for c in train_df.columns if c not in drop_cols and c != label_col]
    common_cols = [c for c in feat_cols if c in test_df.columns]
    if len(common_cols) != len(feat_cols):
        logger.warning("Feature columns differ between train and test; aligning to intersection")
    X_all = train_df[common_cols].copy()
    y_all = train_df[label_col].copy()
    X_test = test_df[common_cols].copy()
    y_test = test_df[label_col].copy()

    # Cast selected categorical columns to int (already mapped)
    cat_cols = _resolve_cat_features(train_df[common_cols + [label_col]], label_col, args.cat_cols, args.cat_max_card)
    for c in cat_cols:
        if c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors='coerce').fillna(-1).astype(np.int32)
            X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(-1).astype(np.int32)

    # Split validation from train
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=args.val_frac, random_state=args.random_seed,
        stratify=y_all if y_all.nunique() > 1 else None
    )

    # Label encode to 0..K-1
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_.tolist()
    num_class = int(len(class_names))
    logger.info(f"[+] Classes ({num_class}): {class_names}")

    # Choose implementation
    impl = args.impl
    if impl == 'hist':
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except Exception as e:
            raise SystemExit(f"scikit-learn with HistGradientBoostingClassifier is required for --impl hist. Error: {e}")
        # Configure threads via env vars if requested
        if int(args.n_threads) > 0:
            os.environ['OMP_NUM_THREADS'] = str(int(args.n_threads))
            # os.environ['OPENBLAS_NUM_THREADS'] = os.environ.get('OPENBLAS_NUM_THREADS', str(int(args.n_threads)))
            # os.environ['MKL_NUM_THREADS'] = os.environ.get('MKL_NUM_THREADS', str(int(args.n_threads)))
            # os.environ['NUMEXPR_NUM_THREADS'] = os.environ.get('NUMEXPR_NUM_THREADS', str(int(args.n_threads)))
            logger.info(f"[+] Using HistGradientBoosting with OMP_NUM_THREADS={args.n_threads}")
        hb = HistGradientBoostingClassifier(
            learning_rate=float(args.learning_rate),
            max_iter=int(args.n_estimators),
            max_depth=(None if int(args.max_depth) <= 0 else int(args.max_depth)),
            random_state=int(args.random_seed),
            validation_fraction=float(args.val_frac),
            n_iter_no_change=int(args.early_stopping),
            verbose=100,
        )
        logger.info(f"[+] Training HistGradientBoosting (iters={args.n_estimators}, lr={args.learning_rate}) …")
        hb.fit(X_tr, y_tr_enc)
        val_score = float(hb.score(X_val, y_val_enc))
        logger.info(f"[+] Validation accuracy: {val_score:.4f}")
        model_impl = hb
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gbm = GradientBoostingClassifier(
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            subsample=float(args.subsample),
            random_state=int(args.random_seed)
        )
        logger.info(f"[+] Training GradientBoosting (n_estimators={args.n_estimators}, lr={args.learning_rate}) …")
        gbm.fit(X_tr, y_tr_enc)
        val_score = float(gbm.score(X_val, y_val_enc))
        logger.info(f"[+] Validation accuracy: {val_score:.4f}")
        model_impl = gbm

    # Predict
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    logger.info("[+] Evaluating on test …")
    y_pred_enc = model_impl.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average='macro'))
    prem = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recm = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Save model and metrics
    try:
        import joblib
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        joblib.dump(model_impl, args.model_out)
        logger.info(f"[+] Saved model -> {args.model_out}")
    except Exception as e:
        logger.warning(f"[!] Failed to save model: {e}")

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump({
            'accuracy': acc,
            'f1_macro': f1m,
            'precision_macro': prem,
            'recall_macro': recm,
            'classes': class_names,
            'features': list(X_all.columns),
        }, f, indent=2)
    logger.info(f"[+] Test accuracy={acc:.4f}, macro-F1={f1m:.4f}, macro-Precision={prem:.4f}, macro-Recall={recm:.4f}")
    logger.info(f"[+] Metrics JSON -> {args.report_out}")

    # Confusion matrix (prefer original label names if encoders available)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        try:
            from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
            pre = CIC2018Preprocessor()
            if not pre.load_encoders():
                raise RuntimeError("encoders not found")
            le_global = pre.encoders.get('label')
            y_test_orig = le_global.inverse_transform(np.asarray(y_test))
            y_pred_orig = le_global.inverse_transform(np.asarray(y_pred))
            labels_sorted = sorted(np.unique(np.concatenate([y_test_orig, y_pred_orig])))
            cm_mat = confusion_matrix(y_test_orig, y_pred_orig, labels=labels_sorted)
        except Exception:
            labels_sorted = class_names
            cm_mat = cm
        logger.info("Confusion matrix (rows=true, cols=pred):")
        header = "labels," + ",".join(map(str, labels_sorted))
        logger.info(header)
        for i, row in enumerate(cm_mat):
            logger.info(f"{labels_sorted[i]}," + ",".join(str(int(x)) for x in row))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_mat, display_labels=labels_sorted)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(include_values=True, cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
        os.makedirs(os.path.dirname(args.cm_out), exist_ok=True)
        plt.tight_layout()
        fig.savefig(args.cm_out, dpi=150)
        plt.close(fig)
        logger.info(f"[+] Confusion matrix PNG -> {args.cm_out}")
    except Exception as e:
        logger.warning(f"[!] Failed to plot confusion matrix: {e}")


if __name__ == "__main__":
    main()





