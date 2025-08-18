import os
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from resampling.undersampling.enn_refiner import ENNRefiner


logger = get_logger(__name__)


ENCODED_DIR = cic2018.ENCODED_DATA_FOLDER
RAW_DIR = cic2018.RAW_PROCESSED_DATA_FOLDER


def find_encoded_files_for_refine(strategy: str) -> list[str]:
    files = []
    if not os.path.isdir(ENCODED_DIR):
        raise SystemExit(f"Encoded directory not found: {ENCODED_DIR}")
    for fname in sorted(os.listdir(ENCODED_DIR)):
        # majority train compressed encoded
        if fname.endswith('_majority_train_compressed_encoded.csv'):
            files.append(os.path.join(ENCODED_DIR, fname))
            continue
        # minority <strategy> train augmented encoded
        suffix = f'_minority_{strategy}_train_augmented_encoded.csv'
        if fname.endswith(suffix):
            files.append(os.path.join(ENCODED_DIR, fname))
    return files


def load_union_encoded(file_paths: list[str]) -> pd.DataFrame:
    frames = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            if 'Label' not in df.columns:
                logger.warning(f"[!] Skip (no Label column): {fp}")
                continue
            frames.append(df)
        except Exception as e:
            logger.warning(f"[!] Failed to load {fp}: {e}")
    if not frames:
        raise SystemExit("No encoded files loaded. Ensure majority compression and minority augmenting have been run.")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Refine combined (major+minor) train with ENN and output single RAW file")
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan', 'cfm', 'fdm'],
                        help='Minority augmenting strategy used (to locate per-label encoded files)')
    parser.add_argument('--tau', type=int, default=20000, help='Final per-class size after refinement')
    parser.add_argument('--n-neighbors', type=int, default=3, help='ENN n_neighbors')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Prepare preprocessor/encoders
    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Run cli.cic2018.encode_data first.")

    # Locate encoded inputs
    files = find_encoded_files_for_refine(args.augmenting_strategy)
    if not files:
        raise SystemExit(f"No encoded inputs found in {ENCODED_DIR} for strategy '{args.augmenting_strategy}'.")
    logger.info(f"[ENN] Found {len(files)} encoded files for refine")

    # Load union encoded
    union_enc = load_union_encoded(files)
    logger.info(f"[ENN] Union encoded shape: {union_enc.shape}; classes={union_enc['Label'].nunique()}")

    # Run ENN refinement across all classes
    refiner = ENNRefiner(tau_final=args.tau, n_neighbors=args.n_neighbors)
    refined_enc = refiner.refine(union_enc, label_col='Label')
    logger.info(f"[ENN] Refined encoded shape: {refined_enc.shape}")

    # Inverse to RAW
    refined_raw = pre.inverse_transform(refined_enc)
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"cic2018_train_refined_{args.augmenting_strategy}_raw.csv")
    refined_raw.to_csv(out_path, index=False)
    logger.info(f"[ENN] Saved refined RAW: {out_path} ({len(refined_raw)})")


if __name__ == '__main__':
    main()


