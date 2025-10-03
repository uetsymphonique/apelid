import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

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
            # Try enable GPU predictor if requested
            try:
                if device in ('cuda', 'GPU', 'auto') and device != 'cpu':
                    # sklearn API
                    if hasattr(model.model, 'set_params'):
                        try:
                            model.model.set_params(**{'device': 'cuda'})
                        except Exception:
                            pass
                    # native Booster API
                    if hasattr(model.model, 'set_param'):
                        try:
                            model.model.set_param({'device': 'cuda'})
                        except Exception:
                            pass
                    logger.info("[Eval][XGB] Using GPU (device=cuda) when available")
                else:
                    logger.info("[Eval][XGB] Using CPU predictor")
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
                    # CatBoost inference GPU depends on build; we log intent
                    if hasattr(model.model, 'set_param'):
                        try:
                            model.model.set_param({'task_type': 'GPU'})
                        except Exception:
                            pass
                    logger.info("[Eval][CatBoost] GPU requested; proceeding if supported by build")
                else:
                    logger.info("[Eval][CatBoost] Using CPU")
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


def _evaluate_single_model(model, model_type: str, X_test: np.ndarray, y_test: np.ndarray, 
                          class_names: list, output_dir: str, model_name: str, 
                          save_metrics: bool = False, save_cm: bool = False):
    """Evaluate a single model and save results."""
    logger.info(f"=== Evaluating {model_name} ===")
    
    # Predict
    y_pred_idx = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluate_classification(y_test, y_pred_idx, class_names)
    
    # Save metrics if requested
    if save_metrics:
        metrics_file = os.path.join(output_dir, model_type, f"{model_name}_metrics.json")
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[+] Metrics saved to: {metrics_file}")
    else:
        # Print metrics to debug log if not saving
        logger.debug(f"{model_name} metrics: {json.dumps(metrics, indent=2)}")
    
    # Generate and save confusion matrix if requested
    if save_cm:
        cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
        cm_file = os.path.join(output_dir, model_type, f"{model_name}_cm.png")
        os.makedirs(os.path.dirname(cm_file), exist_ok=True)
        save_confusion_matrix_png(cm_mat, class_names, cm_file)
        logger.info(f"[+] Confusion matrix saved to: {cm_file}")
    else:
        # Print confusion matrix to debug log if not saving
        cm_mat = confusion_matrix_from_indices(y_test, y_pred_idx, num_classes=len(class_names))
        logger.debug(f"{model_name} confusion matrix:")
        logger.debug(f"Labels: {class_names}")
        logger.debug(f"Matrix:\n{cm_mat}")
    
    logger.info(f"[+] {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
    
    return metrics


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
    parser.add_argument('--mode', type=str, required=True, choices=['all', 'model'], 
                       help="Evaluation mode: 'all' for all found models, 'model' for specific models")
    parser.add_argument('--models', type=str, nargs='*', default=[], 
                       help="Specific models to evaluate (when mode=model)")
    parser.add_argument('--exclude-models', type=str, nargs='*', default=[], 
                       help="Models to exclude from evaluation")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--data-in', type=str, default=None, help="Input test data file")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory for results")
    parser.add_argument('--save-cm', action='store_true', help="Save confusion matrix for each model")
    parser.add_argument('--save-mt', action='store_true', help="Save metrics for each model")
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
        args.output_dir = os.path.join(res.REPORT_FOLDER, 'evaluation')
    
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
    if args.mode == 'all':
        model_files = _find_model_files(args.models_dir, res.resources_name)
        if not model_files:
            raise SystemExit(f"No model files found in {args.models_dir}")
    else:  # mode == 'model'
        if not args.models:
            raise SystemExit("Must specify --models when mode=model")
        model_files = {}
        for model_type in args.models:
            if model_type not in MODEL_CLASSES:
                raise SystemExit(f"Unknown model type: {model_type}")
            ext = '.pth' if model_type == 'dnn' else '.pkl'
            model_path = os.path.join(args.models_dir, f"{res.resources_name}_{model_type}{ext}")
            if os.path.exists(model_path):
                model_files[model_type] = model_path
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    # Apply exclusions
    for exclude_model in args.exclude_models:
        if exclude_model in model_files:
            del model_files[exclude_model]
            logger.info(f"[+] Excluded model: {exclude_model}")
    
    if not model_files:
        raise SystemExit("No models to evaluate after exclusions")
    
    logger.info(f"[+] Evaluating {len(model_files)} models: {list(model_files.keys())}")
    
    # Prepare test data
    # Use tree mode for all models (DNN will be handled separately if needed)
    X_test, y_test, meta_test = PrepareData.prepare_input_data(
        df_test, pre, encode_numerical=False, include_label=True, mode='tree'
    )
    class_names = config_class_names
    
    # Results storage
    all_results = {}
    
    # Evaluate each model
    for model_type, model_path in model_files.items():
        try:
            # Load model
            logger.debug(f"[+] Loading {model_type} model from {model_path}")
            
            if model_type == 'dnn':
                # For DNN, we need to prepare data with scaling
                X_test_dnn, _, meta_test_dnn = PrepareData.prepare_input_data(
                    df_test, pre, encode_numerical=True, include_label=True, mode='dnn'
                )
                model = _load_model(model_path, model_type, 
                                  num_class=len(class_names), 
                                  input_dim=X_test_dnn.shape[1],
                                  device=args.device)
                X_eval = X_test_dnn
            else:
                model = _load_model(model_path, model_type, num_class=len(class_names), device=args.device)
                X_eval = X_test
            
            # Evaluate
            metrics = _evaluate_single_model(
                model, model_type, X_eval, y_test, class_names, 
                args.output_dir, f"{res.resources_name}_{model_type}",
                save_metrics=args.save_mt, save_cm=args.save_cm
            )
            all_results[model_type] = metrics
            
        except Exception as e:
            logger.error(f"[!] Failed to evaluate {model_type}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_file = os.path.join(args.output_dir, f"{res.resources_name}_all_models_metrics.json")
        os.makedirs(os.path.dirname(combined_file), exist_ok=True)
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"[+] Combined results saved to: {combined_file}")
        
        # Print summary
        logger.info("[+] Evaluation Summary:")
        for model_type, metrics in all_results.items():
            logger.info(f"  {model_type}: Accuracy={metrics['accuracy']:.4f}, F1-macro={metrics['f1_macro']:.4f}")
    else:
        logger.warning("[!] No models were successfully evaluated")


if __name__ == "__main__":
    main()
