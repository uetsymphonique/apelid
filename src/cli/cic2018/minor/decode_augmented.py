import os
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


def _default_in_path(label: str, strategy: str) -> str:
    safe = cic2018.get_label_name(label)
    in_dir = os.path.join(ENCODED_DIR, 'train')
    return os.path.join(in_dir, f"cic2018_{safe}_minority_{strategy}_train_augmented_encoded.csv")


def _default_out_path(label: str, strategy: str) -> str:
    safe = cic2018.get_label_name(label)
    out_dir = os.path.join(RAW_DIR, 'train')
    return os.path.join(out_dir, f"cic2018_{safe}_minority_{strategy}_train_augmented_raw_processed.csv")


def main():
    parser = argparse.ArgumentParser(description="Decode augmented minority train sets (encoded → raw_processed) per label")
    parser.add_argument('--strategy', '-a', type=str, default='wgan', choices=['wgan', 'cfm', 'fdm'],
                        help='Augmenting strategy used to build input filenames (default: wgan)')
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Decode all minority labels or a provided list')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MINORITY_LABELS,
                        help='List of labels to decode when mode=label')
    parser.add_argument('--numerical-inverse', type=str, default='quantile_uniform', choices=['quantile_normal', 'standard', 'quantile_uniform', 'minmax'],
                        help='Method to inverse transform numerical features')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Override input directory (defaults to encoded/train)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory (defaults to raw_processed/train)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        target_labels = args.labels
    else:
        target_labels = cic2018.MINORITY_LABELS

    logger.info(f"[+] Decode augmented minority train | strategy={args.strategy} | labels={target_labels}")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Fit encoders first: python -m cli.cic2018.preprocessing.setup_encoders")

    for label in target_labels:
        # Resolve paths
        in_path = _default_in_path(label, args.strategy) if args.input_dir is None else \
            os.path.join(args.input_dir, f"cic2018_{cic2018.get_label_name(label)}_minority_{args.strategy}_train_augmented_encoded.csv")
        out_path = _default_out_path(label, args.strategy) if args.out_dir is None else \
            os.path.join(args.out_dir, f"cic2018_{cic2018.get_label_name(label)}_minority_{args.strategy}_train_augmented_raw_processed.csv")

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


