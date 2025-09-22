import os
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
CLEAN_MERGED_DIR = cic2018.CLEAN_MERGED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


def _select_label_files(input_dir: str, allowed_safe_labels: set[str], subset: str) -> list[str]:
    """Return paths to per-label clean merged CSVs filtered by allowed safe labels and subset."""
    files: list[str] = []
    if subset == 'full':
        scan_dir = input_dir
        suffix = '_clean_merged.csv'
    else:
        scan_dir = os.path.join(input_dir, subset)
        suffix = f'_{subset}_clean_merged.csv'

    if not os.path.isdir(scan_dir):
        return []

    for fname in sorted(os.listdir(scan_dir)):
        if not fname.endswith(suffix):
            continue
        base = fname
        if not base.startswith('cic2018_'):
            continue
        label_safe = base[len('cic2018_'):-len(suffix)]
        if label_safe in allowed_safe_labels:
            files.append(os.path.join(scan_dir, fname))
    return files


def main():
    parser = argparse.ArgumentParser(description="Encode per-label datasets for MINORITY classes (no PCA/UMAP): one-hot + MinMax/Quantile")
    parser.add_argument('--input-dir', type=str, default=CLEAN_MERGED_DIR,
                        help='Input directory containing per-label clean merged CSVs')
    parser.add_argument('--num-encoder', '-n', type=str, default='quantile_uniform', choices=['minmax', 'quantile_uniform'],
                        help='Numerical encoder to use for minority encoding')
    parser.add_argument('--subset', type=str, default='full', choices=['full', 'train', 'test'],
                        help="Which subset of clean-merged to encode: full (unsplit), train or test")
    parser.add_argument('--mode', '-m', type=str, default='all', choices=['all', 'label'],
                        help='Encode all minority labels or a provided list')
    parser.add_argument('--labels', '-c', type=str, nargs='+', default=None, choices=cic2018.MINORITY_LABELS,
                        help='List of label names to encode when mode=label')
    parser.add_argument('--exclude-labels', '-x', type=str, nargs='+', default=None, choices=cic2018.MINORITY_LABELS,
                        help='Labels to exclude from encoding (applied after --mode/--labels)')
    parser.add_argument('--precheck', action='store_true', help='Precheck for duplicates')
    parser.add_argument('--postcheck', action='store_true', help='Postcheck for duplicates')
    parser.add_argument('--sentinel-impute', type=str, default='none', choices=['median', 'zero', 'none'],
                        help='Handle -1 sentinel in Init Fwd/Bwd Win Byts before scaling (default zero; options: median/zero/none) with indicator columns')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    logger.info("[+] ===========================================")
    logger.info("[+] MINORITY ENCODING PIPELINE STARTED")
    logger.info("[+] ===========================================")
    logger.info(f"[+] Numerical encoder: {args.num_encoder}; subset={args.subset}")

    # Resolve target labels
    if args.mode == 'label':
        if not args.labels:
            raise SystemExit("--labels is required when --mode label")
        target_labels = args.labels
    else:
        target_labels = cic2018.MINORITY_LABELS

    # Apply excludes if provided
    if args.exclude_labels:
        excludes = set(args.exclude_labels)
        target_labels = [lbl for lbl in target_labels if lbl not in excludes]
        logger.info(f"[+] Excluding labels: {sorted(excludes)}")

    if not target_labels:
        raise SystemExit("No labels to encode after applying excludes.")

    logger.info(f"[+] Target labels: {target_labels}")

    allowed_safe = {cic2018.get_label_name(lbl) for lbl in target_labels}

    preprocessor = CIC2018Preprocessor()
    if not preprocessor.load_encoders():
        raise SystemExit("Encoders not found. Fit encoders on TRAIN first: python -m cli.cic2018.preprocessing.setup_encoders")

    def process_subset(subset: str):
        label_files = _select_label_files(input_dir, allowed_safe, subset)
        if not label_files:
            logger.info(f"[+] Skip subset={subset}: no files found")
            return

        # Resolve output directories based on subset
        if subset == 'full':
            encoded_out_dir = ENCODED_DIR
        else:
            encoded_out_dir = os.path.join(ENCODED_DIR, subset)
        os.makedirs(encoded_out_dir, exist_ok=True)

        logger.info(f"[+] Found {len(label_files)} minority label files (subset={subset})")
        logger.info("[+] Encoding and exporting MINORITY per-label datasets (encoded only, no PCA/UMAP)...")
        encoded_counts: dict[str, int] = {}

        for lf in label_files:
            base = os.path.basename(lf)
            suffix = '_clean_merged.csv' if subset == 'full' else f'_{subset}_clean_merged.csv'
            label_safe = base[len('cic2018_'):-len(suffix)]
            logger.info(f"[+] Encoding {base} -> label={label_safe} (subset={subset})")

            df_label = pd.read_csv(lf, low_memory=False)

            # Handle sentinel -1 columns before scaling (adds *_is_missing)
            if args.sentinel_impute != 'none':
                fill_val = 0.0 if args.sentinel_impute == 'zero' else 0.0
                df_label = preprocessor.add_sentinel_indicators_and_impute_init_win_bytes(
                    df_label, strategy='median' if args.sentinel_impute == 'median' else 'fixed', fill_value=fill_val
                )

            # Numerical encode (minmax or quantile_uniform)
            if args.num_encoder == 'quantile_uniform':
                enc_df = preprocessor.preprocess_encode_numerical_features_quantile_uniform(df_label.copy())
            else:
                enc_df = preprocessor.preprocess_encode_numerical_features(df_label.copy())

            # Binary and label/categorical encoding (one-hot/ordinal as configured)
            enc_df = preprocessor.preprocess_encode_binary_features(enc_df)
            enc_df = preprocessor.preprocess_encode_label(enc_df)
            enc_df = preprocessor.preprocess_encode_categorical_features(enc_df)

            # Export encoded per-label
            enc_fname = f"cic2018_{label_safe}_encoded.csv"
            enc_path = os.path.join(encoded_out_dir, enc_fname)
            preprocessor.export_encoded_data(enc_df, enc_path)
            encoded_counts[label_safe] = len(enc_df)
            logger.info(f"[+] Saved encoded: {enc_path} ({len(enc_df)})")

        logger.info("[+] ===========================================")
        logger.info(f"[+] MINORITY ENCODING COMPLETED (subset={subset})")
        logger.info("[+] ===========================================")
        logger.info(f"[+] Encoded outputs -> {encoded_out_dir}")
        if encoded_counts:
            sample_items = list(encoded_counts.items())[:15]
            logger.info("[+] Sample encoded counts:")
            for name, cnt in sample_items:
                logger.info(f"    {name}: {cnt}")
            if len(encoded_counts) > 15:
                logger.info(f"    ... and {len(encoded_counts) - 15} more labels")

    # If subset=full, process both train and test sequentially when available
    if args.subset == 'full':
        for sub in ['train', 'test']:
            process_subset(sub)
    else:
        process_subset(args.subset)


if __name__ == "__main__":
    main()


