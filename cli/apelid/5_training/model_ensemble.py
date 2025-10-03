import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logging import setup_logging, get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from preprocessing.prepare import PrepareData

from training.xgb import XGBModel
from training.dnn import DNNModel
from training.catb import CatBoostModel
from training.bagging import BaggingModel
from training.histgbm import HistGBMModel
from training.model import evaluate_classification, confusion_matrix_from_indices, save_confusion_matrix_png


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_CLASSES = {
    'xgb': XGBModel,
    'dnn': DNNModel,
    'catb': CatBoostModel,
    'bagging': BaggingModel,
    'histgbm': HistGBMModel,
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgb': 0.3,
    'dnn': 0.1,
    'catb': 0.2,
    'bagging': 0.2,
    'histgbm': 0.2,
}


def _load_model(model_path: str, model_type: str, num_class: int = None, input_dim: int = None, device: str = 'auto'):
    """Load a trained model from file."""
    if model_type == 'dnn':
        # Load PyTorch state_dict
        map_location = 'cpu' if device == 'cpu' else None
        state_dict = torch.load(model_path, map_location=map_location)
        model = DNNModel(
            input_dim=input_dim,
            num_class=num_class,
            lr=1e-3,  # Default values, will be overridden by state_dict
            weight_decay=1e-4,
            batch_size=512,
            max_epochs=50,
            patience=8,
            device=None if device == 'auto' else device,
            random_state=42,
        )
        model.model.load_state_dict(state_dict)
        model._is_fitted = True
        return model
    else:
        # Load tree models with pickle
        with open(model_path, 'rb') as f:
            model_obj = pickle.load(f)
        
        if model_type == 'xgb':
            model = XGBModel(num_class=num_class, params={}, num_round=100, early_stopping=20, random_state=42)
            model.model = model_obj
            # Enable GPU predictor if requested
            try:
                if device in ('cuda', 'GPU', 'auto') and device != 'cpu':
                    if hasattr(model.model, 'set_params'):
                        try:
                            model.model.set_params(**{'device': 'cuda'})
                        except Exception:
                            pass
                    if hasattr(model.model, 'set_param'):
                        try:
                            model.model.set_param({'device': 'cuda'})
                        except Exception:
                            pass
                    logger.info("[Ensemble][XGB] Using GPU (device=cuda) when available")
                else:
                    logger.info("[Ensemble][XGB] Using CPU predictor")
            except Exception:
                pass
            model._is_fitted = True
            return model
        elif model_type == 'catb':
            model = CatBoostModel(num_class=num_class, params={}, random_state=42)
            model.model = model_obj
            # Try enable GPU if requested
            try:
                if device in ('cuda', 'GPU', 'auto') and device != 'cpu':
                    if hasattr(model.model, 'set_param'):
                        try:
                            model.model.set_param({'task_type': 'GPU'})
                        except Exception:
                            pass
                    logger.info("[Ensemble][CatBoost] GPU requested; proceeding if supported by build")
                else:
                    logger.info("[Ensemble][CatBoost] Using CPU")
            except Exception:
                pass
            model._is_fitted = True
            return model
        elif model_type == 'bagging':
            model = BaggingModel(params={}, random_state=42)
            model.model = model_obj
            model._is_fitted = True
            return model
        elif model_type == 'histgbm':
            model = HistGBMModel(num_class=num_class, params={}, random_state=42)
            model.model = model_obj
            model._is_fitted = True
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def _find_model_files(models_dir: str, resource_name: str) -> dict:
    """Find all model files in the models directory."""
    model_files = {}
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return model_files
    
    for model_type in MODEL_CLASSES.keys():
        # Look for .pth (DNN) or .pkl (tree models)
        ext = '.pth' if model_type == 'dnn' else '.pkl'
        pattern = f"{resource_name}_{model_type}{ext}"
        model_path = models_dir / pattern
        
        if model_path.exists():
            model_files[model_type] = str(model_path)
            logger.debug(f"[+] Found {model_type} model: {model_path}")
        else:
            logger.warning(f"[!] Model not found: {model_path}")
    
    return model_files


def _predict_single_model(model_path: str, model_type: str, X_test: np.ndarray, 
                         num_class: int, input_dim: int = None, device: str = 'auto'):
    """Predict probabilities for a single model."""
    try:
        logger.debug(f"[+] Loading {model_type} model from {model_path}")
        
        if model_type == 'dnn':
            # For DNN, we need to prepare data with scaling
            model = _load_model(model_path, model_type, 
                              num_class=num_class, 
                              input_dim=input_dim,
                              device=device)
            # X_test should already be preprocessed for DNN
            proba = model.predict_proba(X_test)
        else:
            model = _load_model(model_path, model_type, num_class=num_class)
            proba = model.predict_proba(X_test)
        
        if proba is None:
            logger.warning(f"[!] {model_type} does not support predict_proba, using predict")
            pred = model.predict(X_test)
            # Convert predictions to one-hot probabilities
            proba = np.zeros((len(pred), num_class))
            proba[np.arange(len(pred)), pred] = 1.0
        
        logger.debug(f"[+] {model_type} prediction shape: {proba.shape}")
        return model_type, proba
        
    except Exception as e:
        logger.error(f"[!] Failed to predict with {model_type}: {e}")
        return model_type, None


def _predict_with_batching(model_path: str, model_type: str, X_test: np.ndarray, 
                         num_class: int, input_dim: int = None, device: str = 'auto',
                         batch_size: int = -1, model: object = None):
    """Predict with batching for large datasets."""
    if batch_size == -1 or X_test.shape[0] <= batch_size:
        if model is not None:
            # Use pre-loaded model
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                else:
                    pred = model.predict(X_test)
                    proba = np.zeros((len(pred), num_class))
                    proba[np.arange(len(pred)), pred] = 1.0
                return model_type, proba
            except Exception as e:
                logger.error(f"[!] Failed to predict with pre-loaded {model_type}: {e}")
                return model_type, None
        else:
            return _predict_single_model(model_path, model_type, X_test, num_class, input_dim, device)
    
    logger.info(f"[+] Predicting {model_type} in batches of {batch_size}")
    all_proba = []
    
    # Load model once if not provided
    if model is None:
        try:
            model = _load_model(model_path, model_type, num_class, input_dim, device)
        except Exception as e:
            logger.error(f"[!] Failed to load {model_type} model: {e}")
            return model_type, None
    
    for i in range(0, X_test.shape[0], batch_size):
        batch_end = min(i + batch_size, X_test.shape[0])
        X_batch = X_test[i:batch_end]
        
        logger.debug(f"[+] Processing batch {i//batch_size + 1}/{(X_test.shape[0] + batch_size - 1)//batch_size}")
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_batch)
            else:
                pred = model.predict(X_batch)
                proba = np.zeros((len(pred), num_class))
                proba[np.arange(len(pred)), pred] = 1.0
            
            if proba is not None:
                all_proba.append(proba)
            else:
                logger.warning(f"[!] Batch {i//batch_size + 1} failed for {model_type}")
                return model_type, None
        except Exception as e:
            logger.error(f"[!] Batch {i//batch_size + 1} error for {model_type}: {e}")
            return model_type, None
    
    if all_proba:
        final_proba = np.vstack(all_proba)
        logger.info(f"[+] {model_type} batch prediction completed: {final_proba.shape}")
        return model_type, final_proba
    else:
        return model_type, None


def _ensemble_predict(model_files: dict, X_test: np.ndarray, X_test_dnn: np.ndarray, 
                     num_class: int, input_dim: int, max_workers: int = 4, device: str = 'auto',
                     sequential: bool = False, batch_size: int = -1):
    """Ensemble prediction with optional sequential processing and batching."""
    logger.info(f"[+] Ensemble prediction with {len(model_files)} models (sequential={sequential}, batch_size={batch_size})")
    
    results = {}
    
    if sequential:
        # Sequential processing - load models once and reuse
        for model_type, model_path in model_files.items():
            logger.info(f"[+] Processing {model_type} sequentially")
            
            # Load model once
            try:
                if model_type == 'dnn':
                    model = _load_model(model_path, model_type, num_class, input_dim, device)
                    _, proba = _predict_with_batching(model_path, model_type, X_test_dnn, 
                                                    num_class, input_dim, device, batch_size, model)
                else:
                    model = _load_model(model_path, model_type, num_class, None, device)
                    _, proba = _predict_with_batching(model_path, model_type, X_test, 
                                                    num_class, None, device, batch_size, model)
                
                if proba is not None:
                    results[model_type] = proba
                    logger.info(f"[+] {model_type} completed: {proba.shape}")
                else:
                    logger.warning(f"[!] {model_type} failed")
                    
            except Exception as e:
                logger.error(f"[!] Failed to load/process {model_type}: {e}")
                continue
    else:
        # Parallel processing - each worker loads its own model
        tasks = []
        for model_type, model_path in model_files.items():
            if model_type == 'dnn':
                tasks.append((model_path, model_type, X_test_dnn, num_class, input_dim, device, batch_size))
            else:
                tasks.append((model_path, model_type, X_test, num_class, None, device, batch_size))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(_predict_with_batching, *task): task[1] 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    model_type_result, proba = future.result()
                    if proba is not None:
                        results[model_type_result] = proba
                        logger.debug(f"[+] {model_type_result} prediction completed")
                    else:
                        logger.warning(f"[!] {model_type_result} prediction failed")
                except Exception as e:
                    logger.error(f"[!] {model_type} prediction error: {e}")
    
    if not results:
        raise RuntimeError("No models produced valid predictions")
    
    # Weighted ensemble
    logger.info(f"[+] Combining predictions from {len(results)} models")
    ensemble_proba = np.zeros((X_test.shape[0], num_class))
    total_weight = 0.0
    
    for model_type, proba in results.items():
        weight = ENSEMBLE_WEIGHTS.get(model_type, 0.0)
        if weight > 0:
            # Check if proba shape matches expected num_class
            if proba.shape[1] != num_class:
                logger.warning(f"[!] {model_type} has {proba.shape[1]} classes, expected {num_class}. Skipping.")
                continue
            
            ensemble_proba += weight * proba
            total_weight += weight
            logger.debug(f"[+] Added {model_type} with weight {weight:.2f} (shape: {proba.shape})")
    
    if total_weight > 0:
        ensemble_proba /= total_weight
        logger.info(f"[+] Ensemble prediction completed (total weight: {total_weight:.2f})")
    else:
        logger.warning("[!] No valid weights found, using uniform voting")
        ensemble_proba = np.mean(list(results.values()), axis=0)
    
    # Convert to predictions
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    return ensemble_proba, ensemble_pred


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction using multiple trained models")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], 
                       help="Models to exclude from ensemble")
    parser.add_argument('--max-workers', type=int, default=4, help="Maximum parallel workers")
    parser.add_argument('--sequential', action='store_true', help="Run models sequentially instead of parallel")
    parser.add_argument('--batch-size', type=int, default=-1, help="Batch size for prediction (-1 for full dataset)")
    parser.add_argument('--save-cm', action='store_true', help="Save confusion matrix")
    parser.add_argument('--save-mt', action='store_true', help="Save metrics")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Setup resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()
    
    # Get class names from configs (not from encoder)
    all_labels = res.MAJORITY_LABELS + res.MINORITY_LABELS
    config_class_names = sorted(all_labels)
    
    # Default paths
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')
    if args.output_dir is None:
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'ensemble')
    
    # Default test data path
    if args.data_in is None:
        _, default_test = _default_inputs(res)
        args.data_in = default_test
    
    # Load data
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")
    
    logger.info(f"[+] Loading test data: {args.data_in}")
    df_test = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Test data shape: {df_test.shape}")
    
    # Load encoders
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")
    
    # Find model files
    model_files = _find_model_files(args.models_dir, res.resources_name)
    if not model_files:
        raise SystemExit(f"No model files found in {args.models_dir}")
    
    # Apply exclusions
    for exclude_model in args.exclude_models:
        if exclude_model in model_files:
            del model_files[exclude_model]
            logger.info(f"[+] Excluded model: {exclude_model}")
    
    if not model_files:
        raise SystemExit("No models to ensemble after exclusions")
    
    logger.info(f"[+] Ensemble with {len(model_files)} models: {list(model_files.keys())}")
    
    # Prepare test data
    # Tree mode for most models
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, encode_numerical=False, include_label=True, mode='tree'
    )
    class_names = config_class_names
    
    # DNN mode for DNN model
    X_test_dnn, _, meta_test_dnn = PrepareData.prepare_input_data(
        df_test, pre, encode_numerical=True, include_label=True, mode='dnn'
    )
    input_dim = X_test_dnn.shape[1]
    
    # Ensemble prediction
    try:
        ensemble_proba, ensemble_pred = _ensemble_predict(
            model_files, X_test, X_test_dnn, 
            len(class_names), input_dim, args.max_workers, args.device,
            sequential=args.sequential, batch_size=args.batch_size
        )
        
        # Calculate metrics
        metrics = evaluate_classification(y_test, ensemble_pred, class_names)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics if requested
        if args.save_mt:
            metrics_file = os.path.join(args.output_dir, f"{res.resources_name}_ensemble_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[+] Metrics saved to: {metrics_file}")
        else:
            logger.debug(f"Ensemble metrics: {json.dumps(metrics, indent=2)}")
        
        # Save confusion matrix if requested
        if args.save_cm:
            cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
            cm_file = os.path.join(args.output_dir, f"{res.resources_name}_ensemble_cm.png")
            save_confusion_matrix_png(cm_mat, class_names, cm_file)
            logger.info(f"[+] Confusion matrix saved to: {cm_file}")
        else:
            cm_mat = confusion_matrix_from_indices(y_test, ensemble_pred, num_classes=len(class_names))
            logger.debug(f"Ensemble confusion matrix:")
            logger.debug(f"Labels: {class_names}")
            logger.debug(f"Matrix:\n{cm_mat}")
        
        # Print summary
        logger.info(f"[+] Ensemble - Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
        logger.info(f"[+] Ensemble weights: {ENSEMBLE_WEIGHTS}")
        
    except Exception as e:
        logger.error(f"[!] Ensemble prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
