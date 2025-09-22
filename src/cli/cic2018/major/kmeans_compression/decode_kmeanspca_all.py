import os
import argparse
import pandas as pd

from utils.logging import setup_logging, get_logger
from configs import cic2018
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor


logger = get_logger(__name__)


MAJOR_LABELS = cic2018.MAJORITY_LABELS


def _paths(label: str) -> tuple[str, str, str, str]:
    safe = cic2018.get_label_name(label)
    enc_tr = os.path.join(cic2018.ENCODED_DATA_FOLDER, 'train', f"cic2018_{safe}_encoded_compressed_kmeanspca_all.csv")
    enc_te = os.path.join(cic2018.ENCODED_DATA_FOLDER, 'test',  f"cic2018_{safe}_encoded_selected_kmeanspca_all.csv")
    raw_tr = os.path.join(cic2018.RAW_PROCESSED_DATA_FOLDER, 'train', f"cic2018_{safe}_raw_processed_compressed_kmeanspca_all.csv")
    raw_te = os.path.join(cic2018.RAW_PROCESSED_DATA_FOLDER, 'test',  f"cic2018_{safe}_raw_processed_selected_kmeanspca_all.csv")
    return enc_tr, enc_te, raw_tr, raw_te


def _decode_file(enc_path: str, out_path: str) -> None:
    if not os.path.exists(enc_path):
        logger.warning(f"Missing encoded input: {enc_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_enc = pd.read_csv(enc_path, low_memory=False)
    try:
        pre = CIC2018Preprocessor()
        pre.load_encoders()
        df_raw = pre.inverse_transform(df_enc, numerical_inverse='quantile_normal')
        df_raw.to_csv(out_path, index=False)
        logger.info(f"Decoded -> {out_path} ({len(df_raw)})")
    except Exception as e:
        logger.error(f"Failed to decode {enc_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Decode kmeanspca_all encoded outputs to raw_processed")
    parser.add_argument('--label', type=str, required=True, choices=MAJOR_LABELS + ['All'])
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    labels = MAJOR_LABELS if args.label == 'All' else [args.label]
    for lb in labels:
        try:
            enc_tr, enc_te, raw_tr, raw_te = _paths(lb)
            _decode_file(enc_tr, raw_tr)
            _decode_file(enc_te, raw_te)
        except Exception as e:
            logger.warning(f"[{lb}] failed: {e}")


if __name__ == "__main__":
    main()


