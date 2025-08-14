import os
import pandas as pd

from utils.logging import get_logger, setup_logging
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from datasvc.data_service import DataService


logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
FINAL_TRAIN = os.path.join(DATA_FOLDER, "cic_final_train_balanced.csv")
FINAL_TEST  = os.path.join(DATA_FOLDER, "cic_final_test.csv")

OUT_TRAIN = os.path.join(DATA_FOLDER, "cic_final_train_balanced_cat_map.csv")
OUT_TEST  = os.path.join(DATA_FOLDER, "cic_final_test_cat_map.csv")


def main():
    setup_logging("INFO")

    if not (os.path.exists(FINAL_TRAIN) and os.path.exists(FINAL_TEST)):
        raise SystemExit(
            "Final datasets not found. Run 'python -m cli-cic.merge_final' first."
        )

    logger.info("[+] Loading final train/test …")
    train_df = pd.read_csv(FINAL_TRAIN)
    test_df = pd.read_csv(FINAL_TEST)
    logger.info(f"[+] Train: {train_df.shape}, Test: {test_df.shape}")

    pre = CIC2018Preprocessor()
    if not pre.load_encoders():
        logger.info("[+] Encoders not found, fitting on train set …")
        # Ensure selection matches training features
        train_df = pre.select_features_and_label(train_df)
        test_df = pre.select_features_and_label(test_df)
        pre.setup_encoders(train_df)
        pre.save_encoders()
    else:
        # Ensure only expected features are present
        train_df = pre.select_features_and_label(train_df)
        test_df = pre.select_features_and_label(test_df)

    # Ordinal-encode categorical features (Protocol) for classical models
    train_df_mapped = pre.preprocess_encode_ordinal_features(train_df.copy())
    test_df_mapped  = pre.preprocess_encode_ordinal_features(test_df.copy())

    # Optional: do not encode labels here (keep original strings) for readability
    # train_df_mapped = pre.preprocess_encode_label(train_df_mapped)
    # test_df_mapped  = pre.preprocess_encode_label(test_df_mapped)

    # Drop duplicates
    train_svc = DataService(train_df_mapped)
    test_svc  = DataService(test_df_mapped)
    train_svc.fix_duplicates()
    test_svc.fix_duplicates()

    # Brief categorical stats
    cat_feats = pre.encoded_categorical_features_ordinal
    merged = pd.concat([train_svc.df, test_svc.df], ignore_index=True)
    logger.info("categorical features details (merged train and test):")
    for col in cat_feats:
        if col in merged.columns:
            logger.info(f"{col}: {merged[col].nunique()} unique; min={merged[col].min()}, max={merged[col].max()}")

    # Export
    train_svc.export_data(OUT_TRAIN)
    test_svc.export_data(OUT_TEST)
    logger.info(f"[+] Saved: {OUT_TRAIN}")
    logger.info(f"[+] Saved: {OUT_TEST}")


if __name__ == "__main__":
    main()


