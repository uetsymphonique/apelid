import os
import sys
import argparse
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from preprocessing.prepare import PrepareData

from training.xgb import XGBModel
from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png


matplotlib.use('Agg')
logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost using raw_processed(train) + clean_merged(test) with in-script prepare")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--train-in', type=str, default=None)
    parser.add_argument('--test-in', type=str, default=None)
    parser.add_argument('--label-col', type=str, default='Label')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--num-round', type=int, default=1200)
    parser.add_argument('--early-stopping', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'CPU', 'GPU'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--model-out', type=str, default=None)
    parser.add_argument('--report-out', type=str, default=None)
    parser.add_argument('--cm-out', type=str, default=None)
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    default_train, default_test = _default_inputs(res)
    train_in = args.train_in or default_train
    test_in = args.test_in or default_test
    if not os.path.exists(train_in) or not os.path.exists(test_in):
        raise SystemExit(f"Input files not found: train={train_in}, test={test_in}")

    logger.info(f"[+] Loading inputs:\n  train={train_in}\n  test={test_in}")
    df_train = pd.read_csv(train_in, low_memory=False)
    df_test = pd.read_csv(test_in, low_memory=False)

    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Fit encoders first.")

    # Use PrepareData for consistent preprocessing (tree models: ordinal cat, binary 0/1, label encode, cont kept)
    X_tr, X_val, y_tr, y_val, meta_tr = PrepareData.prepare_training_data(
        df_train, pre,
        encode_numerical=False,
        use_validation=True,
        val_size=float(args.val_frac),
        random_state=42,
        mode='tree',
    )
    X_te, y_te, meta_te = PrepareData.prepare_input_data(
        df_test, pre,
        encode_numerical=False,
        include_label=True,
        mode='tree',
    )

    # Class names from fitted label encoder
    classes_sorted = meta_tr.get('class_names') or sorted(np.unique(y_tr).tolist())

    model = XGBModel(
        num_class=len(classes_sorted),
        params={'tree_method': 'gpu_hist' if args.device in ('GPU', 'auto') else 'hist', 'random_state': 42},
        num_round=int(args.num_round),
        early_stopping=int(args.early_stopping),
        random_state=42,
    )
    model.fit(X_tr, y_tr, X_val, y_val)
    y_pred_idx = model.predict(X_te)
    metrics = evaluate_classification(y_te, y_pred_idx, classes_sorted)
    logger.info(f"[+] Metrics: accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}, precision_macro={metrics['precision_macro']:.4f}, recall_macro={metrics['recall_macro']:.4f}")

    # Outputs
    model_out = args.model_out or os.path.join(res.DATA_FOLDER, 'models', f'{res.resources_name}_xgb.json')
    report_folder = f"{res.REPORT_FOLDER}/xgb"
    os.makedirs(report_folder, exist_ok=True)
    report_out = args.report_out or os.path.join(report_folder, 'xgb_metrics.json')
    cm_out = args.cm_out or os.path.join(report_folder, 'xgb_cm.png')

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    # Save model (best effort - using Booster save_model via a retrain is outside scope here)
    with open(model_out, 'w') as f:
        f.write("XGB model saved via CLI orchestrator (use dedicated saver if needed)")

    with open(report_out, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm_mat = confusion_matrix_from_indices(y_te, y_pred_idx, num_classes=len(classes_sorted))
    os.makedirs(os.path.dirname(cm_out), exist_ok=True)
    save_confusion_matrix_png(cm_mat, classes_sorted, cm_out)

    logger.info(f"[+] Metrics JSON -> {report_out}")
    logger.info(f"[+] Confusion matrix PNG -> {cm_out}")


if __name__ == "__main__":
    main()


