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


def _resolve_cat_features(df: pd.DataFrame, label_col: str, user_cols: list[str] | None) -> list[str]:
    if user_cols:
        return [c for c in user_cols if c in df.columns and c != label_col]
    # Default: known mapped categorical columns if present
    defaults = ["Protocol", "Dst Port"]
    return [c for c in defaults if c in df.columns and c != label_col]


def _cast_cats_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(-1).astype(np.int32)
    return out


def _build_model(params: dict, device: str, gpu_devices: str):
    from catboost import CatBoostClassifier
    cfg = dict(
        iterations=int(params['iterations']),
        depth=int(params['depth']),
        learning_rate=float(params['learning_rate']),
        l2_leaf_reg=float(params['l2_leaf_reg']),
        random_seed=int(params['random_seed']),
        loss_function='MultiClass',
        eval_metric='TotalF1',
        od_type='Iter',
        od_wait=100,
        verbose=100,
        auto_class_weights=None,
    )
    if device == 'GPU':
        cfg['task_type'] = 'GPU'
        cfg['devices'] = gpu_devices
    else:
        cfg['task_type'] = 'CPU'
    return CatBoostClassifier(**cfg)


def _prob_matrix(model, pool) -> np.ndarray:
    # Returns array (N, K) with probabilities
    proba = model.predict(pool, prediction_type='Probability')
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.stack([1 - proba, proba], axis=1)
    return proba


def _tune_threshold(y_true: np.ndarray,
                    proba: np.ndarray,
                    class_names: list[str],
                    label_benign: str,
                    label_infil: str,
                    mode: str = 'margin',
                    opt_metric: str = 'pair_macro_f1') -> tuple[float, dict]:
    from sklearn.metrics import f1_score
    # indices
    idx_b = class_names.index(label_benign)
    idx_i = class_names.index(label_infil)

    thresholds = np.linspace(-0.4, 0.4, num=81) if mode == 'margin' else np.linspace(0.2, 0.9, num=71)
    best_tau = 0.0
    best_score = -1.0

    # baseline preds
    base_idx = np.argmax(proba, axis=1)
    base_pred = np.array([class_names[i] for i in base_idx])

    for tau in thresholds:
        pred = base_pred.copy()
        p_b = proba[:, idx_b]
        p_i = proba[:, idx_i]
        if mode == 'margin':
            margin = p_i - p_b
            # if model says Benign but margin>=tau, switch to Infiltration
            mask = (base_pred == label_benign) & (margin >= tau)
        else:  # prob
            mask = (base_pred == label_benign) & (p_i >= tau)
        pred[mask] = label_infil

        if opt_metric == 'infil_f1':
            f1 = f1_score((y_true == label_infil), (pred == label_infil))
        else:
            # macro F1 over the pair {B, I}
            y_pair = y_true.copy()
            y_pair[(y_pair != label_benign) & (y_pair != label_infil)] = 'Other'
            pred_pair = pred.copy()
            pred_pair[(pred_pair != label_benign) & (pred_pair != label_infil)] = 'Other'
            # compute macro-F1 for B vs I only (drop Other)
            mask_bi = (y_pair != 'Other')
            if mask_bi.any():
                f1_b = f1_score((y_pair[mask_bi] == label_benign), (pred_pair[mask_bi] == label_benign))
                f1_i = f1_score((y_pair[mask_bi] == label_infil), (pred_pair[mask_bi] == label_infil))
                f1 = 0.5 * (f1_b + f1_i)
            else:
                f1 = 0.0

        if f1 > best_score:
            best_score = f1
            best_tau = float(tau)

    return best_tau, {'best_score': float(best_score), 'mode': mode, 'opt_metric': opt_metric}


def main():
    try:
        from catboost import Pool
    except Exception as e:
        raise SystemExit(f"CatBoost is required. Please install: pip install catboost. Error: {e}")

    parser = argparse.ArgumentParser(description="Train CatBoost with class weights for Infilteration and tune Benign↔Infiltration threshold")
    parser.add_argument('--train-in', type=str, default=_default_inputs()[0])
    parser.add_argument('--test-in', type=str, default=_default_inputs()[1])
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--drop-cols', type=str, nargs='*', default=['__source__'])
    parser.add_argument('--cat-cols', type=str, nargs='*', default=None)
    # Class weights
    parser.add_argument('--benign-weight', type=float, default=1.0)
    parser.add_argument('--infil-weight', type=float, default=4.0)
    parser.add_argument('--other-weight', type=float, default=1.0)
    # Device/params
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'CPU', 'GPU'])
    parser.add_argument('--gpu-devices', type=str, default='0')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.08)
    parser.add_argument('--l2-leaf-reg', type=float, default=3.0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--val-frac', type=float, default=0.1)
    # Threshold
    parser.add_argument('--tune-mode', type=str, default='margin', choices=['margin', 'prob'])
    parser.add_argument('--opt-metric', type=str, default='pair_macro_f1', choices=['pair_macro_f1', 'infil_f1'])
    # Outputs
    parser.add_argument('--model-out', type=str, default=os.path.join(cic2018.DATA_FOLDER, 'models', 'cic2018_catboost_wt.cbm'))
    parser.add_argument('--report-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'metrics_wt.json'))
    parser.add_argument('--cm-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'confusion_matrix_wt.png'))
    parser.add_argument('--threshold-out', type=str, default=os.path.join(cic2018.REPORT_FOLDER, 'catboost', 'threshold.json'))
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load data
    for p in [args.train_in, args.test_in]:
        if not os.path.exists(p):
            raise SystemExit(f"Input not found: {p}")
    train_df = pd.read_csv(args.train_in, low_memory=False)
    test_df = pd.read_csv(args.test_in, low_memory=False)
    label_col = args.label_col

    drop_cols = [c for c in (args.drop_cols or []) if c in train_df.columns]
    feat_cols = [c for c in train_df.columns if c not in drop_cols and c != label_col]
    common_cols = [c for c in feat_cols if c in test_df.columns]
    if len(common_cols) != len(feat_cols):
        logger.warning("Feature columns differ between train and test; aligning to intersection")

    X_all = train_df[common_cols].copy()
    y_all = train_df[label_col].copy()
    X_test = test_df[common_cols].copy()
    y_test = test_df[label_col].copy()

    # Cats
    cat_cols = _resolve_cat_features(train_df[common_cols + [label_col]], label_col, args.cat_cols)
    X_all = _cast_cats_to_int(X_all, cat_cols)
    X_test = _cast_cats_to_int(X_test, cat_cols)
    cat_idx = [X_all.columns.get_loc(c) for c in cat_cols]
    logger.info(f"[+] Categorical features: {cat_cols} (idx={cat_idx})")

    # Split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=args.val_frac, random_state=args.random_seed,
        stratify=y_all if y_all.nunique() > 1 else None
    )

    # Class names and weights
    class_names = sorted(y_tr.unique().tolist())
    weights = {lbl: args.other_weight for lbl in class_names}
    if 'Benign' in weights:
        weights['Benign'] = args.benign_weight
    if 'Infilteration' in weights:
        weights['Infilteration'] = args.infil_weight
    class_weights = [float(weights[lbl]) for lbl in class_names]

    # Pools
    from catboost import Pool
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx if cat_idx else None)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx if cat_idx else None)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx if cat_idx else None)

    # Build model (GPU if available/requested)
    params = dict(iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
                  l2_leaf_reg=args.l2_leaf_reg, random_seed=args.random_seed)
    device = args.device
    model = None
    if device in ('GPU', 'auto'):
        try:
            logger.info("[+] Trying GPU …")
            model = _build_model(params, 'GPU', args.gpu_devices)
            model.set_params(class_names=class_names, class_weights=class_weights)
        except Exception as e:
            logger.warning(f"[!] GPU init failed: {e}")
            model = None
    if model is None:
        logger.info("[+] Using CPU …")
        model = _build_model(params, 'CPU', args.gpu_devices)
        model.set_params(class_names=class_names, class_weights=class_weights)

    # Fit
    logger.info(f"[+] Training with class_weights: {dict(zip(class_names, class_weights))}")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Tune threshold on val
    proba_val = _prob_matrix(model, val_pool)
    y_val_arr = y_val.values.astype(object)
    tau, tune_info = _tune_threshold(y_val_arr, proba_val, class_names, 'Benign', 'Infilteration', args.tune_mode, args.opt_metric)
    logger.info(f"[+] Tuned threshold τ={tau:.4f} (mode={tune_info['mode']}, metric={tune_info['opt_metric']}, score={tune_info['best_score']:.4f})")

    # Evaluate on test with postprocess
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    proba_test = _prob_matrix(model, test_pool)
    idx_top = np.argmax(proba_test, axis=1)
    pred_raw = np.array([class_names[i] for i in idx_top])
    # Apply threshold rewrite Benign→Infiltration
    idx_b = class_names.index('Benign') if 'Benign' in class_names else None
    idx_i = class_names.index('Infilteration') if 'Infilteration' in class_names else None
    pred_adj = pred_raw.copy()
    if idx_b is not None and idx_i is not None:
        p_b = proba_test[:, idx_b]
        p_i = proba_test[:, idx_i]
        if args.tune_mode == 'margin':
            margin = p_i - p_b
            mask = (pred_adj == 'Benign') & (margin >= tau)
        else:
            mask = (pred_adj == 'Benign') & (p_i >= tau)
        pred_adj[mask] = 'Infilteration'

    y_true = y_test.values.astype(object)
    def _metrics(y, yhat):
        return dict(
            accuracy=float(accuracy_score(y, yhat)),
            f1_macro=float(f1_score(y, yhat, average='macro')),
            precision_macro=float(precision_score(y, yhat, average='macro', zero_division=0)),
            recall_macro=float(recall_score(y, yhat, average='macro', zero_division=0)),
        )

    m_raw = _metrics(y_true, pred_raw)
    m_adj = _metrics(y_true, pred_adj)
    logger.info(f"[+] Test RAW    | acc={m_raw['accuracy']:.4f} f1={m_raw['f1_macro']:.4f} pre={m_raw['precision_macro']:.4f} rec={m_raw['recall_macro']:.4f}")
    logger.info(f"[+] Test THRESH | acc={m_adj['accuracy']:.4f} f1={m_adj['f1_macro']:.4f} pre={m_adj['precision_macro']:.4f} rec={m_adj['recall_macro']:.4f}")

    # Log and plot CM for adjusted
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        labels_sorted = sorted(np.unique(np.concatenate([y_true, pred_adj])))
        cm_mat = confusion_matrix(y_true, pred_adj, labels=labels_sorted)
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

    # Save model and threshold
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save_model(args.model_out)
    os.makedirs(os.path.dirname(args.threshold_out), exist_ok=True)
    with open(args.threshold_out, 'w') as f:
        json.dump({'tau': float(tau), 'mode': args.tune_mode, 'opt_metric': args.opt_metric,
                   'class_names': class_names, 'class_weights': dict(zip(class_names, map(float, class_weights)))}, f, indent=2)

    # Save metrics
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump({'raw': m_raw, 'thresholded': m_adj}, f, indent=2)

    logger.info(f"[+] Saved model -> {args.model_out}")
    logger.info(f"[+] Saved threshold -> {args.threshold_out}")
    logger.info(f"[+] Metrics JSON -> {args.report_out}")


if __name__ == "__main__":
    main()


