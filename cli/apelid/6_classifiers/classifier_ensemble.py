import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png  # noqa: E402

from art_classifier import DNNClassifier, SkleanWrapper, CatBoostWrapper, XGBWrapper  # noqa: E402


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm']

# Default ensemble weights (can be adjusted)
ENSEMBLE_WEIGHTS = {
    'xgb': 0.3,
    'dnn': 0.1,
    'catb': 0.2,
    'bagging': 0.2,
    'histgbm': 0.2,
}


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _find_model_files(models_dir: str, resource_name: str) -> dict:
    model_files = {}
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return model_files
    for mt in MODEL_TYPES:
        ext = '.pth' if mt == 'dnn' else '.pkl'
        p = models_dir / f"{resource_name}_{mt}{ext}"
        if p.exists():
            model_files[mt] = str(p)
            logger.debug(f"[+] Found {mt} model: {p}")
        else:
            logger.warning(f"[!] Model not found: {p}")
    return model_files


def _load_wrapper(model_type: str, model_path: str, *, num_classes: int, input_dim_tree: int, input_dim_dnn: int, device: str):
    if model_type == 'dnn':
        return DNNClassifier.from_checkpoint(model_path, input_dim=input_dim_dnn, num_classes=num_classes, device=device)

    with open(model_path, 'rb') as f:
        model_obj = pickle.load(f)

    if model_type in ('bagging', 'histgbm'):
        return SkleanWrapper(model=model_obj, num_classes=num_classes, input_shape=(input_dim_tree,), device='cpu')

    if model_type == 'catb':
        try:
            return CatBoostWrapper(model=model_obj, num_classes=num_classes, input_shape=(input_dim_tree,), device=device)
        except Exception as e:
            logger.warning(f"CatBoostWrapper unavailable, fallback to SkleanWrapper: {e}")
            return SkleanWrapper(model=model_obj, num_classes=num_classes, input_shape=(input_dim_tree,), device='cpu')

    if model_type == 'xgb':
        try:
            return XGBWrapper(model=model_obj, num_classes=num_classes, input_shape=(input_dim_tree,), device=device)
        except Exception as e:
            logger.warning(f"XGBWrapper not compatible with saved model, skipping: {e}")
            raise

    raise ValueError(f"Unknown model type: {model_type}")


def _predict_one(wrapper, model_type: str, X: np.ndarray, num_class: int) -> np.ndarray | None:
    try:
        proba = wrapper.predict_proba(X)
        if proba is None:
            pred = wrapper.predict(X)
            proba = np.zeros((len(pred), num_class), dtype=float)
            proba[np.arange(len(pred)), pred] = 1.0
        return proba
    except Exception as e:
        logger.error(f"[!] {model_type} prediction failed: {e}")
        return None


def _predict_with_batching(wrapper, model_type: str, X: np.ndarray, num_class: int, batch_size: int) -> np.ndarray | None:
    if batch_size == -1 or X.shape[0] <= batch_size:
        return _predict_one(wrapper, model_type, X, num_class)

    logger.info(f"[+] Predicting {model_type} in batches of {batch_size}")
    outs = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        proba = _predict_one(wrapper, model_type, batch, num_class)
        if proba is None:
            return None
        outs.append(proba)
    return np.vstack(outs) if outs else None


def _ensemble_predict(model_files: dict, X_tree: np.ndarray, X_dnn: np.ndarray, *, num_class: int, input_dim_dnn: int, input_dim_tree: int,
                      device: str = 'cpu', batch_size: int = -1, max_workers: int = 4, sequential: bool = False):
    logger.info(f"[+] Ensemble prediction with {len(model_files)} models (sequential={sequential}, batch_size={batch_size})")

    # Load wrappers
    wrappers = {}
    for mt, path in model_files.items():
        try:
            w = _load_wrapper(mt, path, num_classes=num_class, input_dim_tree=input_dim_tree, input_dim_dnn=input_dim_dnn, device=device)
            wrappers[mt] = w
        except Exception:
            logger.warning(f"[!] Skip {mt} due to incompatible wrapper")

    if not wrappers:
        raise RuntimeError("No compatible wrappers loaded")

    # Predict
    results = {}
    if sequential:
        for mt, w in wrappers.items():
            X = X_dnn if mt == 'dnn' else X_tree
            proba = _predict_with_batching(w, mt, X, num_class, batch_size)
            if proba is not None:
                results[mt] = proba
                logger.debug(f"[+] {mt} prediction done: {proba.shape}")
    else:
        tasks = []
        for mt, w in wrappers.items():
            X = X_dnn if mt == 'dnn' else X_tree
            tasks.append((w, mt, X, num_class, batch_size))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2mt = {ex.submit(_predict_with_batching, *t): t[1] for t in tasks}
            for fut in as_completed(fut2mt):
                mt = fut2mt[fut]
                try:
                    proba = fut.result()
                    if proba is not None:
                        results[mt] = proba
                except Exception as e:
                    logger.error(f"[!] {mt} prediction error: {e}")

    if not results:
        raise RuntimeError("No models produced valid predictions")

    # Weighted ensemble
    logger.info(f"[+] Combining predictions from {len(results)} models")
    ensemble = np.zeros((X_tree.shape[0], num_class), dtype=float)
    total_w = 0.0
    for mt, proba in results.items():
        w = ENSEMBLE_WEIGHTS.get(mt, 0.0)
        if w <= 0:
            continue
        if proba.shape[1] != num_class:
            logger.warning(f"[!] {mt} classes mismatch: {proba.shape[1]} vs {num_class}, skipping")
            continue
        ensemble += w * proba
        total_w += w
    if total_w > 0:
        ensemble /= total_w
    else:
        logger.warning("[!] No valid weights, using uniform mean")
        ensemble = np.mean(list(results.values()), axis=0)

    y_pred = np.argmax(ensemble, axis=1)
    return ensemble, y_pred


def main():
    parser = argparse.ArgumentParser(description="ART-based ensemble prediction")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], help="Models to exclude from ensemble")
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--save-cm', action='store_true')
    parser.add_argument('--save-mt', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    class_names = sorted(all_labels)

    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'classifier_ensemble')

    if args.data_in is None:
        _, default_test = _default_inputs(res)
        args.data_in = default_test

    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")

    logger.info(f"[+] Loading test data: {args.data_in}")
    df_test = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Test data shape: {df_test.shape}")

    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    model_files = _find_model_files(args.models_dir, res.resources_name)
    for ex in args.exclude_models:
        if ex in model_files:
            del model_files[ex]
            logger.info(f"[+] Excluded model: {ex}")
    if not model_files:
        raise SystemExit("No models to ensemble after exclusions")
    logger.info(f"[+] Ensemble with {len(model_files)} models: {list(model_files.keys())}")

    # Features for tree and DNN
    X_tree, y_test, _ = PrepareData.prepare_input_data(df_test, pre, encode_numerical=False, include_label=True, mode='tree')
    X_dnn, _, _ = PrepareData.prepare_input_data(df_test, pre, encode_numerical=True, include_label=True, mode='dnn')

    try:
        proba, y_pred = _ensemble_predict(
            model_files,
            X_tree,
            X_dnn,
            num_class=len(class_names),
            input_dim_dnn=X_dnn.shape[1],
            input_dim_tree=X_tree.shape[1],
            device=args.device,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            sequential=args.sequential,
        )

        metrics = evaluate_classification(y_test, y_pred, class_names)

        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_mt:
            metrics_file = os.path.join(args.output_dir, f"{res.resources_name}_art_ensemble_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[+] Metrics saved to: {metrics_file}")
        else:
            logger.debug(f"Ensemble metrics: {json.dumps(metrics, indent=2)}")

        if args.save_cm:
            cm_mat = confusion_matrix_from_indices(y_test, y_pred, num_classes=len(class_names))
            cm_file = os.path.join(args.output_dir, f"{res.resources_name}_art_ensemble_cm.png")
            save_confusion_matrix_png(cm_mat, class_names, cm_file)
            logger.info(f"[+] Confusion matrix saved to: {cm_file}")
        else:
            cm_mat = confusion_matrix_from_indices(y_test, y_pred, num_classes=len(class_names))
            logger.debug("Ensemble confusion matrix:")
            logger.debug(f"Labels: {class_names}")
            logger.debug(f"Matrix:\n{cm_mat}")

        logger.info(f"[+] Ensemble - Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
        logger.info(f"[+] Ensemble weights: {ENSEMBLE_WEIGHTS}")

    except Exception as e:
        logger.error(f"[!] Ensemble prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()


