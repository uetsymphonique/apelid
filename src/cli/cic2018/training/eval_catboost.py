import os
import argparse
import json
import pandas as pd
import numpy as np

from utils.logging import setup_logging, get_logger
from configs import cic2018


logger = get_logger(__name__)


def _default_inputs() -> tuple[str, str, str]:
    data_dir = cic2018.DATA_FOLDER
    model_path = os.path.join(data_dir, 'models', 'cic2018_catboost.cbm')
    test_path = os.path.join(data_dir, 'cic2018_final_test_cat_map.csv')
    report_out = os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'metrics_eval.json')
    return model_path, test_path, report_out


def main():
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as e:
        raise SystemExit(f"CatBoost is required. Please install: pip install catboost. Error: {e}")

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
    
    cat_cols = [c for c in (args.cat_cols or ['Protocol', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt']) if c in X.columns]
    for c in cat_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(-1).astype(np.int32)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    pool = Pool(X, label=y, cat_features=cat_idx if cat_idx else None)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
    logger.info("[+] Predicting …")
    y_pred = model.predict(pool)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    acc = float(accuracy_score(y, y_pred))
    f1m = float(f1_score(y, y_pred, average='weighted'))
    prem = float(precision_score(y, y_pred, average='weighted', zero_division=0))
    recm = float(recall_score(y, y_pred, average='weighted', zero_division=0))
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    # Log CM
    logger.info("Confusion matrix (rows=true, cols=pred):")
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
            'f1_weighted': f1m,
            'precision_weighted': prem,
            'recall_weighted': recm,
            'classification_report': report,
            'features': list(X.columns),
            'cat_features': cat_cols,
        }, f, indent=2)

    # Plot CM
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
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

    logger.info(f"[+] Eval done | acc={acc:.4f} f1_weighted={f1m:.4f} precision_weighted={prem:.4f} recall_weighted={recm:.4f}")


if __name__ == "__main__":
    main()


