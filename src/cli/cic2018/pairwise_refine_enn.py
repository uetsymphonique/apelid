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


def find_encoded_files(strategy: str) -> list[str]:
    files = []
    if not os.path.isdir(ENCODED_DIR):
        raise SystemExit(f"Encoded directory not found: {ENCODED_DIR}")
    for fname in sorted(os.listdir(ENCODED_DIR)):
        # Majority train compressed encoded
        if fname.endswith('_majority_train_compressed_encoded.csv'):
            files.append(os.path.join(ENCODED_DIR, fname))
            continue
        # Minority <strategy> train augmented encoded
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
    parser = argparse.ArgumentParser(description="Pairwise-then-global ENN refine pipeline")
    parser.add_argument('--augmenting-strategy', '-a', type=str, required=True, choices=['wgan', 'cfm', 'fdm'],
                        help='Minority augmenting strategy used (to locate per-label encoded files)')
    parser.add_argument('--class-a', type=str, default='Benign', help='First class name for pairwise strong ENN')
    parser.add_argument('--class-b', type=str, default='Infilteration', help='Second class name for pairwise strong ENN')
    parser.add_argument('--tau', type=int, default=20000, help='Final per-class size after refinement')
    parser.add_argument('--pair-n-neighbors', type=int, default=7, help='Strong ENN k for pairwise step')
    parser.add_argument('--global-n-neighbors', type=int, default=3, help='ENN k for global step')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        raise SystemExit("Encoders not found. Run cli.cic2018.encode_data first.")

    files = find_encoded_files(args.augmenting_strategy)
    if not files:
        raise SystemExit(f"No encoded inputs found in {ENCODED_DIR} for strategy '{args.augmenting_strategy}'.")
    logger.info(f"[PAIR-ENN] Found {len(files)} encoded files")

    union_enc = load_union_encoded(files)
    logger.info(f"[PAIR-ENN] Union encoded: {union_enc.shape}, classes={union_enc['Label'].nunique()}")

    # Map class names to ids
    try:
        a_id = int(pre.encoders['label'].transform([args.class_a])[0])
        b_id = int(pre.encoders['label'].transform([args.class_b])[0])
    except Exception as e:
        raise SystemExit(f"Failed to map class names to ids: {e}")

    # Pairwise strong ENN on A ∪ B
    pair_df = union_enc[union_enc['Label'].isin([a_id, b_id])].copy()
    other_df = union_enc[~union_enc['Label'].isin([a_id, b_id])].copy()
    logger.info(f"[PAIR-ENN] Pair subset sizes: {args.class_a}={len(pair_df[pair_df['Label']==a_id])}, {args.class_b}={len(pair_df[pair_df['Label']==b_id])}")
    ref_pair = ENNRefiner(tau_final=args.tau, n_neighbors=args.pair_n_neighbors).refine(pair_df, label_col='Label')
    logger.info(f"[PAIR-ENN] After pairwise ENN: {ref_pair.shape}")

    # Global ENN on combined
    combined = pd.concat([ref_pair, other_df], ignore_index=True)
    logger.info(f"[PAIR-ENN] Combined before global ENN: {combined.shape}, classes={combined['Label'].nunique()}")
    ref_all = ENNRefiner(tau_final=args.tau, n_neighbors=args.global_n_neighbors).refine(combined, label_col='Label')
    logger.info(f"[PAIR-ENN] After global ENN: {ref_all.shape}, classes={ref_all['Label'].nunique()}")

    # Inverse to RAW
    refined_raw = pre.inverse_transform(ref_all)
    os.makedirs(RAW_DIR, exist_ok=True)
    a_safe = cic2018.get_label_name(args.class_a)
    b_safe = cic2018.get_label_name(args.class_b)
    out_name = f"cic2018_train_refined_pair_{a_safe}_{b_safe}_{args.augmenting_strategy}_raw.csv"
    out_path = os.path.join(RAW_DIR, out_name)
    refined_raw.to_csv(out_path, index=False)
    logger.info(f"[PAIR-ENN] Saved refined RAW: {out_path} ({len(refined_raw)})")


if __name__ == '__main__':
    main()


