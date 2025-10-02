import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from utils.logging import setup_logging, get_logger


import sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import cic2018


matplotlib.use('Agg')

logger = get_logger(__name__)


def _default_inputs() -> tuple[str, str, str]:
    data_dir = cic2018.DATA_FOLDER
    model_path = os.path.join(data_dir, 'models', 'cic2018_catboost.cbm')
    test_path = os.path.join(data_dir, 'cic2018_final_test_cat_map.csv')
    report_out = os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'metrics_eval.json')
    return model_path, test_path, report_out


def main():

    parser = argparse.ArgumentParser(description="Load CatBoost model and evaluate on mapped dataset without retraining")
    parser.add_argument('--model-in', type=str, default=_default_inputs()[0])
    parser.add_argument('--data-in', type=str, default=_default_inputs()[1], help='CSV to evaluate on (mapped test by default)')
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--drop-cols', type=str, nargs='*', default=['__source__'])
    parser.add_argument('--cat-cols', type=str, nargs='*', default=None)
    parser.add_argument('--report-out', type=str, default=_default_inputs()[2])
    parser.add_argument('--cm-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'confusion_matrix_eval.png'))
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not os.path.exists(args.model_in) or not os.path.exists(args.data_in):
        raise SystemExit(f"Model or data not found: model={args.model_in}, data={args.data_in}")

    logger.info(f"[+] Loading model: {args.model_in}")
    model = CatBoostClassifier()
    model.load_model(args.model_in)

    logger.info(f"[+] Loading data: {args.data_in}")
    df = pd.read_csv(args.data_in, low_memory=False)
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found")

    drop_cols = [c for c in (args.drop_cols or []) if c in df.columns]
    feat_cols = [c for c in df.columns if c not in drop_cols and c != args.label_col]
    X = df[feat_cols].copy()
    y = df[args.label_col].copy()

    # Cast categorical columns to int (consistent with training)
    
    cat_cols = [c for c in (args.cat_cols or ['RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt']) if c in X.columns]
    for c in cat_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(-1).astype(np.int32)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    pool = Pool(X, label=y, cat_features=cat_idx if cat_idx else None)

    logger.info("[+] Predicting …")
    y_pred = model.predict(pool)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    acc = float(accuracy_score(y, y_pred))
    f1micro = float(f1_score(y, y_pred, average='micro'))
    premicro = float(precision_score(y, y_pred, average='micro', zero_division=0))
    recmicro = float(recall_score(y, y_pred, average='micro', zero_division=0))
    f1macro = float(f1_score(y, y_pred, average='macro'))
    premacro = float(precision_score(y, y_pred, average='macro', zero_division=0))
    recmacro = float(recall_score(y, y_pred, average='macro', zero_division=0))
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    # Log CM (with inverse label names if available)
    logger.info("Confusion matrix (rows=true, cols=pred):")
    try:
        pre = CIC2018Preprocessor()
        if not pre.load_encoders():
            raise RuntimeError("encoders not found")
        le = pre.encoders.get('label')
        y_orig = le.inverse_transform(np.asarray(y))
        y_pred_orig = le.inverse_transform(np.asarray(y_pred, dtype=type(y.iloc[0]) if hasattr(y, 'iloc') else int))
        labels_sorted = sorted(np.unique(np.concatenate([y_orig, y_pred_orig])))
        header = "labels," + ",".join(map(str, labels_sorted))
        logger.info(header)
        cm_mat = confusion_matrix(y_orig, y_pred_orig, labels=labels_sorted)
        for i, row in enumerate(cm_mat):
            logger.info(f"{labels_sorted[i]}," + ",".join(str(int(x)) for x in row))
    except Exception as e:
        logger.warning(f"[!] Could not inverse-transform labels for CM, falling back to numeric: {e}")
        labels_sorted = sorted(np.unique(np.concatenate([y.values, np.array(y_pred)])))
        header = "labels," + ",".join(map(str, labels_sorted))
        logger.info(header)
        cm_mat = confusion_matrix(y, y_pred, labels=labels_sorted)
        for i, row in enumerate(cm_mat):
            logger.info(f"{labels_sorted[i]}," + ",".join(str(int(x)) for x in row))

    # Save metrics
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump({
            'accuracy': acc,
            'f1_micro': f1micro,
            'precision_micro': premicro,
            'recall_micro': recmicro,
            'f1_macro': f1macro,
            'precision_macro': premacro,
            'recall_macro': recmacro,
            'classification_report': report,
            'features': list(X.columns),
            'cat_features': cat_cols,
        }, f, indent=2)

    # Plot CM
    try:
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

    logger.info(f"[+] Eval done | acc={acc:.4f} f1_micro={f1micro:.4f} precision_micro={premicro:.4f} recall_micro={recmicro:.4f}")
    logger.info(f"                              f1_macro={f1macro:.4f} precision_macro={premacro:.4f} recall_macro={recmacro:.4f}")


if __name__ == "__main__":
    print("Starting evaluation...")
    main()


