import os
import pandas as pd
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor
from datasvc.data_service import DataService
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
MERGED_CSV_PATH = os.path.join(DATA_FOLDER, "CIC2018_merged_clean_dedup.csv")


def resolve_input_path():
    """Resolve the input CSV path, preferring deduped version"""
    dedup_path = os.path.join(DATA_FOLDER, "CIC2018_merged_clean_dedup.csv")
    clean_path = os.path.join(DATA_FOLDER, "CIC2018_merged_clean.csv")
    
    if os.path.exists(dedup_path):
        return dedup_path
    elif os.path.exists(clean_path):
        return clean_path
    else:
        raise FileNotFoundError(f"No cleaned CSV found: {dedup_path} or {clean_path}")


def main():
    setup_logging("INFO")
    
    # Resolve input path
    input_csv = resolve_input_path()
    logger.info(f"[+] Loading data from: {input_csv}")
    
    # Load data
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"[+] Loaded data shape: {df.shape}")
    logger.info(f"[+] Original label distribution:")
    logger.info(df['Label'].value_counts())
    
    # Initialize preprocessor
    preprocessor = CIC2018Preprocessor()
    
    # Step 1: Select features and drop unwanted columns
    df = preprocessor.select_features_and_label(df)
    logger.info(f"[+] After feature selection: {df.shape}")
    
    # Step 2: Data cleaning using DataService
    df_svc = DataService(df)
    logger.info(f"[+] Checking for duplicates: {df_svc.check_duplicates()}")
    logger.info(f"[+] Checking for missing values: {df_svc.check_missing_values()}")
    
    # Fix duplicates and missing values
    df_svc.fix_duplicates()
    df_svc.fix_missing_values()
    df = df_svc.df
    
    logger.info(f"[+] After cleaning: {df.shape}")
    logger.info(f"[+] Cleaned label distribution:")
    logger.info(df['Label'].value_counts())
    
    # Step 3: Setup encoders
    preprocessor.setup_encoders(df)
    preprocessor.save_encoders()
    
    # Step 4: Encode data
    logger.info("[+] Encoding data...")
    encoded_df = preprocessor.preprocess_encode_numerical_features(df.copy())
    encoded_df = preprocessor.preprocess_encode_binary_features(encoded_df)
    encoded_df = preprocessor.preprocess_encode_label(encoded_df)
    encoded_df = preprocessor.preprocess_encode_categorical_features(encoded_df)
    
    logger.info(f"[+] Encoded data shape: {encoded_df.shape}")
    logger.info(f"[+] Encoded label distribution:")
    logger.info(encoded_df['Label'].value_counts())
    
    # Step 5: Export encoded data
    encoded_output_path = os.path.join(DATA_FOLDER, "CIC2018_encoded.csv")
    preprocessor.export_encoded_data(encoded_df, encoded_output_path)
    
    # Step 6: Export raw data (after inverse transform)
    raw_df = preprocessor.inverse_transform(encoded_df)
    raw_output_path = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")
    preprocessor.export_raw_data(raw_df, raw_output_path)
    
    # Step 7: Summary
    logger.info("[+] Preprocessing completed!")
    logger.info(f"[+] Files created:")
    logger.info(f"    - {encoded_output_path} (encoded for WGAN)")
    logger.info(f"    - {raw_output_path} (raw for final training)")
    logger.info(f"[+] Feature breakdown:")
    logger.info(f"    - Categorical (OneHot): {len(preprocessor.encoded_categorical_features)} features")
    logger.info(f"    - Binary: {len(preprocessor.binary_features)} features")
    logger.info(f"    - Numerical (MinMax): {len(preprocessor.encoded_numerical_features)} features")
    logger.info(f"    - Total features after encoding: {encoded_df.shape[1] - 1} (excluding Label)")


if __name__ == "__main__":
    main()