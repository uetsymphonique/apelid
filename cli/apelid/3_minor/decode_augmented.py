import os
import sys
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from configs import CIC2018Resources, NSLKDDResources
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor


logger = get_logger(__name__)


REGISTRY = {
    'cic2018': (CIC2018Resources, CIC2018Preprocessor),
    'nslkdd': (NSLKDDResources, NSLKDDPreprocessor),
}


def main():
    parser = argparse.ArgumentParser(description="Decode augmented minority train sets (encoded → raw_processed) per label (multi-resource)")
    parser.add_argument('--resource', '-r', type=str, required=True, choices=list(REGISTRY.keys()))
    parser.add_argument('--strategy', '-a', type=str, default='wgan', choices=['wgan', 'cfm', 'fdm'],
                        help='Augmenting strategy used to build input filenames (default: wgan)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Decode all minority labels or a provided list')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None,
                        help='List of labels to decode when mode=label')
    parser.add_argument('--numerical-inverse', '-n', type=str, required=True,
                        choices=['quantile_uniform', 'minmax'],
                        help='Method to inverse transform numerical features')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Override input directory (defaults to encoded/train)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory (defaults to raw_processed/train)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    ResourcesClass, PreprocessorClass = REGISTRY[args.resource]
    res = ResourcesClass
    pre = PreprocessorClass()

    # Load encoders for inverse transform
    if not pre.load_encoders():
        raise SystemExit(f"Encoders not found. Fit encoders for {res.resources_name} first.")

    # Resolve labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        target_labels = args.labels
        # validate
        valid = set(res.MINORITY_LABELS)
        invalid = [lb for lb in target_labels if lb not in valid]
        if invalid:
            raise SystemExit(f"Invalid labels for {res.resources_name}: {invalid}")
    else:
        target_labels = res.MINORITY_LABELS

    logger.info(f"[+] Decode augmented minority train | resource={res.resources_name} | strategy={args.strategy} | labels={target_labels}")

    # Directories
    encoded_train_dir = res.encoded_dir_for_subset('train') if args.input_dir is None else args.input_dir
    raw_train_dir = os.path.join(res.RAW_PROCESSED_DATA_FOLDER, 'train') if args.out_dir is None else args.out_dir

    for label in target_labels:
        safe = res.get_label_name(label)
        in_path = os.path.join(encoded_train_dir, f"{res.resources_name}_{safe}_minority_{args.strategy}_train_augmented_encoded.csv")
        out_path = os.path.join(raw_train_dir, f"{res.resources_name}_{safe}_minority_{args.strategy}_train_augmented_raw_processed.csv")

        if not os.path.exists(in_path):
            logger.warning(f"[+] Skip {label}: input not found {in_path}")
            continue

        logger.info(f"[+] Decoding {label}: {in_path}")
        enc_df = pd.read_csv(in_path, low_memory=False)
        logger.info(f"[+] Loaded encoded ({label}): rows={len(enc_df)} cols={len(enc_df.columns)}")

        try:
            raw_df = pre.inverse_transform(enc_df, numerical_inverse=args.numerical_inverse)
        except Exception as e:
            logger.error(f"[!] Failed to inverse_transform for {label}: {e}")
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        raw_df.to_csv(out_path, index=False)
        logger.info(f"[+] Saved raw_processed ({label}) -> {out_path} ({len(raw_df)})")


if __name__ == "__main__":
    main()


