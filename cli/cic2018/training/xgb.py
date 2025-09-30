import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import cic2018

matplotlib.use('Agg')

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
    # Heuristic: integer/low-card columns (excluding label) as categorical
    candidates: list[str] = ['RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'ECE Flag Cnt']
    # for col in df.columns:
    #     if col == label_col or col.startswith("__"):
    #         continue
    #     s = df[col]
    #     if pd.api.types.is_integer_dtype(s) or (pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all()):
    #         try:
    #             nunq = int(s.nunique(dropna=True))
    #             if nunq <= max_cardinality:
    #                 candidates.append(col)
    #         except Exception:
    #             pass
    return candidates


def main():

    parser = argparse.ArgumentParser(description="Train XGBoost on merged CIC2018 mapped train/test (no extra encoding)")
    parser.add_argument('--train-in', type=str, default=_default_inputs()[0])
    parser.add_argument('--test-in', type=str, default=_default_inputs()[1])
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--drop-cols', type=str, nargs='*', default=['__source__'])
    parser.add_argument('--cat-cols', type=str, nargs='*', default=None)
    parser.add_argument('--cat-max-card', type=int, default=256)
    parser.add_argument('--val-frac', type=float, default=0.1)
    # XGB params
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'CPU', 'GPU'])
    parser.add_argument('--num-round', type=int, default=1200)
    parser.add_argument('--eta', type=float, default=0.08)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--subsample', type=float, default=0.9)
    parser.add_argument('--colsample-bytree', type=float, default=0.9)
    parser.add_argument('--lambda-l2', type=float, default=1.0)
    parser.add_argument('--alpha-l1', type=float, default=0.0)
    parser.add_argument('--early-stopping', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=42)
    # Outputs
    parser.add_argument('--model-out', type=str, default=os.path.join(cic2018.DATA_FOLDER, 'models', 'cic2018_xgb.json'))
    parser.add_argument('--report-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'xgb', 'metrics.json'))
    parser.add_argument('--cm-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'xgb', 'confusion_matrix.png'))
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

    # Determine categorical columns (already ordinal-mapped) and cast to int
    cat_cols = _resolve_cat_features(train_df[common_cols + [label_col]], label_col, args.cat_cols, args.cat_max_card)
    for c in cat_cols:
        if c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors='coerce').fillna(-1).astype(np.int32)
            X_test[c] = pd.to_numeric(X_test[c], errors='coerce').fillna(-1).astype(np.int32)

    # Split validation from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=args.val_frac, random_state=args.random_seed,
        stratify=y_all if y_all.nunique() > 1 else None
    )

    # Label encode to 0..K-1 for XGB
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_.tolist()
    num_class = int(len(class_names))
    logger.info(f"[+] Classes ({num_class}): {class_names}")

    # DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr_enc)
    dval = xgb.DMatrix(X_val, label=y_val_enc)
    dtest = xgb.DMatrix(X_test, label=y_test_enc)

    # Params
    if args.device == 'GPU' or (args.device == 'auto'):
        tree_method = 'gpu_hist'
    else:
        tree_method = 'hist'
    params = {
        'objective': 'multi:softprob',
        'num_class': num_class,
        'eta': float(args.eta),
        'max_depth': int(args.max_depth),
        'subsample': float(args.subsample),
        'colsample_bytree': float(args.colsample_bytree),
        'lambda': float(args.lambda_l2),
        'alpha': float(args.alpha_l1),
        'tree_method': tree_method,
        'random_state': int(args.random_seed),
        'eval_metric': 'mlogloss',
        'verbose_eval': False,
    }

    # Train
    logger.info(f"[+] Training XGBoost (tree_method={tree_method}) …")
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=int(args.num_round), evals=watchlist,
                      early_stopping_rounds=int(args.early_stopping))

    # Predict
    logger.info("[+] Evaluating on test …")
    proba = model.predict(dtest)
    y_pred_enc = np.argmax(proba, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average='macro'))
    prem = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recm = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Save model and metrics
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save_model(args.model_out)
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

    logger.info(f"[+] Saved model -> {args.model_out}")
    logger.info(f"[+] Test accuracy={acc:.4f}, macro-F1={f1m:.4f}, macro-Precision={prem:.4f}, macro-Recall={recm:.4f}")
    logger.info(f"[+] Metrics JSON -> {args.report_out}")


    # Try to inverse-transform labels via CIC2018Preprocessor label encoder
    try:
        pre = CIC2018Preprocessor()
        if not pre.load_encoders():
            raise RuntimeError("encoders not found")
        le_global = pre.encoders.get('label')
        y_test_orig = le_global.inverse_transform(np.asarray(y_test))
        y_pred_orig = le_global.inverse_transform(np.asarray(y_pred))
        labels_sorted = sorted(np.unique(np.concatenate([y_test_orig, y_pred_orig])))
        cm_mat = confusion_matrix(y_test_orig, y_pred_orig, labels=labels_sorted)
    except Exception as _e:
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


if __name__ == "__main__":
    main()


