import os
import sys
import argparse
import json

import numpy as np
import pandas as pd

from utils.logging import setup_logging, get_logger


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources  # noqa: E402
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor  # noqa: E402
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor  # noqa: E402
from preprocessing.prepare import PrepareData  # noqa: E402
from helpers.ensemble_helper import load_model, evaluate_single_model  # noqa: E402


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}

MODEL_TYPES = ['xgb', 'dnn', 'catb', 'bagging', 'histgbm']
ATTACK_TYPES = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']


def _default_inputs(res) -> tuple[str, str]:
    data_dir = res.BALENCED_DATA_FOLDER
    train_path = os.path.join(data_dir, f"{res.resources_name}_merged_train_raw_processed.csv")
    test_path = os.path.join(data_dir, f"{res.resources_name}_test_random_sample_clean_merged.csv")
    return train_path, test_path


def _resolve_model_path(models_dir: str, resource_name: str, model_type: str) -> str:
    ext = '.pth' if model_type == 'dnn' else '.pkl'
    p = os.path.join(models_dir, f"{resource_name}_{model_type}{ext}")
    if not os.path.exists(p):
        raise SystemExit(f"Model file not found: {p}")
    return p


def _default_adv_path(res, subset: str, model_type: str, attack: str) -> str:
    base_dir = os.path.join(res.DATA_FOLDER, 'adv_samples', subset)
    filename = f"{res.resources_name}_{model_type}_{attack}_adv.csv"
    return os.path.join(base_dir, filename)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on plain and adversarial (test subset) samples")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--model', '-m', type=str, required=True, choices=MODEL_TYPES)
    parser.add_argument('--attack', '-a', type=str, required=False, choices=ATTACK_TYPES,
                        help="Attack name (required if --adv-in not provided)")
    parser.add_argument('--plain-in', type=str, default=None, help="Plain test CSV. Default = dataset default test")
    parser.add_argument('--adv-in', type=str, default=None, help="Adversarial test CSV. Default = adv_samples/test/... per naming convention")
    parser.add_argument('--models-dir', type=str, default=None, help="Directory containing trained models")
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resource and preprocessor
    ResClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResClass
    pre = PreprocessorClass()

    # Defaults
    if args.models_dir is None:
        args.models_dir = os.path.join(res.DATA_FOLDER, 'models')

    # Resolve inputs
    if args.plain_in is None:
        _, default_test = _default_inputs(res)
        args.plain_in = default_test

    if args.adv_in is None:
        if not args.attack:
            raise SystemExit("When --adv-in is not provided, --attack is required to resolve default adversarial CSV path.")
        args.adv_in = _default_adv_path(res, subset='test', model_type=args.model, attack=args.attack)

    if not os.path.exists(args.plain_in):
        raise SystemExit(f"Plain test CSV not found: {args.plain_in}")
    if not os.path.exists(args.adv_in):
        raise SystemExit(f"Adversarial test CSV not found: {args.adv_in}")

    logger.info(f"[+] Plain test CSV: {args.plain_in}")
    logger.info(f"[+] Adversarial test CSV: {args.adv_in}")

    # Load encoders
    if not pre.load_encoders(fixed_label_encoder=True):
        raise SystemExit("Encoders not found. Train models first.")

    # Prepare data (plain)
    df_plain = pd.read_csv(args.plain_in, low_memory=False)
    X_plain, y_plain, meta_plain = PrepareData.prepare_input_data(df_plain, pre, include_label=True)

    # Prepare data (adversarial)
    df_adv = pd.read_csv(args.adv_in, low_memory=False)
    X_adv, y_adv, meta_adv = PrepareData.prepare_input_data(df_adv, pre, include_label=True)

    # Class names from configs (stable order)
    class_names = sorted(res.MAJORITY_LABELS + res.MINORITY_LABELS)

    # Load model
    model_path = _resolve_model_path(args.models_dir, res.resources_name, args.model)
    logger.info(f"[+] Loading model: type={args.model}, path={model_path}")
    model = load_model(model_path, args.model, num_class=len(class_names), device=args.device)

    # Evaluate on plain test
    logger.info("[+] Evaluating on plain test set")
    metrics_plain = evaluate_single_model(
        model, args.model, X_plain, y_plain, class_names,
        output_dir=None, model_name=f"{res.resources_name}_{args.model}_plain",
        save_metrics=False, save_cm=False
    )
    logger.info(f"[Plain] Accuracy={metrics_plain['accuracy']:.4f}, F1-macro={metrics_plain['f1_macro']:.4f}")

    # Evaluate on adversarial test
    logger.info("[+] Evaluating on adversarial test set")
    metrics_adv = evaluate_single_model(
        model, args.model, X_adv, y_adv, class_names,
        output_dir=None, model_name=f"{res.resources_name}_{args.model}_{args.attack or 'adv'}",
        save_metrics=False, save_cm=False
    )
    logger.info(f"[Adversarial] Accuracy={metrics_adv['accuracy']:.4f}, F1-macro={metrics_adv['f1_macro']:.4f}")


if __name__ == "__main__":
    main()


