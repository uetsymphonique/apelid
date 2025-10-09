import os
import sys
import argparse
import pandas as pd

from utils.logging import get_logger, setup_logging



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor

# WGAN
from resampling.data_augmentation.augmented_wgan.pipeline import (
    generate_augmented_samples as wgan_generate,
    AugmentOptions as WGANOptions,
)


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def _encode_like_minor(pre, df: pd.DataFrame, num_encoder: str) -> pd.DataFrame:
    df2 = pre.select_features_and_label(df.copy())
    logger.info(f"[+] Encoding like minority with {num_encoder} encoder")

    if num_encoder == 'quantile_uniform' and hasattr(pre, 'preprocess_encode_numerical_features_quantile_uniform'):
        enc_df = pre.preprocess_encode_numerical_features_quantile_uniform(df2)
    elif num_encoder == 'minmax' and hasattr(pre, 'preprocess_encode_numerical_features_minmax'):
        enc_df = pre.preprocess_encode_numerical_features_minmax(df2)
    else:
        raise ValueError(f"Invalid numerical encoder for opposite sampling: {num_encoder}")

    if hasattr(pre, 'preprocess_encode_binary_features'):
        enc_df = pre.preprocess_encode_binary_features(enc_df)
    else:
        raise AttributeError(f"No compatible binary encoder found on preprocessor")
    if hasattr(pre, 'preprocess_encode_label'):
        enc_df = pre.preprocess_encode_label(enc_df)
    else:
        raise AttributeError(f"No compatible label encoder found on preprocessor")
    if hasattr(pre, 'preprocess_encode_categorical_features'):
        enc_df = pre.preprocess_encode_categorical_features(enc_df)
    else:
        raise AttributeError(f"No compatible categorical encoder found on preprocessor")
    return enc_df


def _load_clean_merged_opposite(res, pre, label_name: str, subset: str = 'train', num_encoder: str = 'quantile_uniform'):
    try:
        safe = res.get_label_name(label_name)
        # Prefer compressed if available (for Benign), else fall back to original clean_merged
        cmc = res.clean_merged_path_for(subset, safe, compressed=True)
        if os.path.exists(cmc):
            df = pd.read_csv(cmc, low_memory=False)
        else:
            cm = res.clean_merged_path_for(subset, safe, compressed=False)
            if not os.path.exists(cm):
                raise FileNotFoundError(f"Opposite clean_merged not found: {cm}")
            df = pd.read_csv(cm, low_memory=False)

        try:
            return _encode_like_minor(pre, df, num_encoder=num_encoder)
        except Exception as e:
            logger.warning(f"[WGAN] Opposite encoding failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"[WGAN] Opposite clean_merged loader failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Minority augmenting (WGAN) for multiple resources")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan'],
                        help='Augmenting strategy to use (currently: wgan)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Augment all predefined minority labels or provided labels')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None,
                        help='Label names to augment when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None,
                        help='Labels to exclude from augmentation')
    parser.add_argument('--opposite-label', type=str, default='Benign',
                        help='Label to use as critic opposite class (default Benign)')
    parser.add_argument('--num-encoder', '-n', type=str, required=True,
                        choices=['minmax', 'quantile_uniform'],
                        help='Numerical encoder for opposite sampling encode (match minority encoding)')
    parser.add_argument('--tau', '-t', type=int, required=True, help='Base target samples per class after augmentation')
    parser.add_argument('--log-level', '-L', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResourcesClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResourcesClass
    pre = PreprocessorClass()

    # Load encoders required by augmentation pipelines
    if not pre.load_encoders():
        raise SystemExit(f"Encoders not found. Fit encoders for {res.resources_name} first.")

    # Resolve labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        minority_labels = args.labels
        # validate
        valid = set(res.MINORITY_LABELS)
        invalid = [lb for lb in minority_labels if lb not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
    else:
        minority_labels = res.MINORITY_LABELS
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        minority_labels = [lbl for lbl in minority_labels if lbl not in excludes]

    logger.info(f"[+] Minority labels to augment ({res.resources_name}): {minority_labels}")

    # Directories and accept rates
    encoded_train_dir = res.encoded_dir_for_subset('train')
    os.makedirs(encoded_train_dir, exist_ok=True)
    accept_rate_map = getattr(res, 'ACCEPT_RATE_MAP', {}) or {}

    # Opposite label id (optional)
    try:
        critic_id = pre.encoders['label'].transform([args.opposite_label])[0]
    except Exception:
        critic_id = None

    for lbl in minority_labels:
        safe = res.get_label_name(lbl)
        enc_train_path = os.path.join(encoded_train_dir, res.encoded_filename_for_label(safe))
        if not os.path.exists(enc_train_path):
            logger.warning(f"[+] Encoded train not found for {lbl}: {enc_train_path}; skip")
            continue
        train_df = pd.read_csv(enc_train_path, low_memory=False)

        # Load encoded test if available (for dedup keys in WGAN)
        encoded_test_dir = res.encoded_dir_for_subset('test')
        enc_test_path = os.path.join(encoded_test_dir, res.encoded_filename_for_label(safe))
        if os.path.exists(enc_test_path):
            test_df = pd.read_csv(enc_test_path, low_memory=False)
            logger.info(f"[+] Loaded encoded test for dedup: {len(test_df)} rows")
        else:
            # Create minimal test_df by sampling from train_df to avoid pipeline issues
            test_df = train_df.sample(n=min(100, len(train_df)), random_state=42).copy()
            logger.info(f"[+] No encoded test found, using train sample as test_df for {lbl} ({len(test_df)} rows)")

        if args.augmenting_strategy == 'wgan':
            # Accept rate per label with sensible fallbacks
            ar = float(accept_rate_map.get(lbl, 0.30))
            opts = WGANOptions(
                use_benign_for_critic=True,
                critic_epochs=60,
                wgan_iterations=10000,
                d_iter=5,
                use_gp=True,
                accept_rate=ar,
                request_multiplier=3.0,
                max_rounds=40,
                use_postfilter=True,
                min_precision=0.95,
                use_encoded_dedup=True,
                trim_to_need=True,
                use_final_fill=True,
            )

            def _benign_loader():
                return _load_clean_merged_opposite(
                    res,
                    pre,
                    label_name=args.opposite_label,
                    subset='train',
                    num_encoder=args.num_encoder,
                )

            aug_train = wgan_generate(
                pre=pre,
                class_name=lbl,
                train_df=train_df,
                test_df=test_df,
                benign_loader=_benign_loader,
                tau=int(args.tau),
                device='auto',
                accept_rate=ar,
                min_precision=0.95,
                options=opts,
                critic_id=critic_id,
            )
        else:
            logger.error(f"[+] Augmenting strategy not implemented: {args.augmenting_strategy}")
            continue

        # Save per-label outputs (encoded train only)
        enc_out = os.path.join(encoded_train_dir, f"{res.resources_name}_{safe}_minority_{args.augmenting_strategy}_train_augmented_encoded.csv")
        aug_train.to_csv(enc_out, index=False)
        logger.info(f"[+] Saved augmented minority train ENCODED ({lbl}): {enc_out} ({len(aug_train)})")

    logger.info(f"[+] {res.resources_name} minority augmenting completed")


if __name__ == "__main__":
    main()


