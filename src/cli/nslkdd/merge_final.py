from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import get_logger, setup_logging
import pandas as pd
from sklearn.utils import shuffle

logger = get_logger(__name__)

def merge_final_datasets():
    """
    Merge majority and minority datasets to create final balanced dataset
    """
    logger.info("[+] Starting final dataset merge...")
    
    # Load majority data
    try:
        majority_train = pd.read_csv('data/majority_train_compressed_raw.csv')
        majority_test = pd.read_csv('data/majority_test_compressed_raw.csv')
        logger.info(f"[+] Loaded majority data: {len(majority_train)} train, {len(majority_test)} test")
    except FileNotFoundError:
        logger.error("[-] Majority data files not found. Run majority_compression.py first.")
        return
    
    # Load minority data
    try:
        minority_train = pd.read_csv('data/minority_train_augmented_raw.csv')
        minority_test = pd.read_csv('data/minority_test_raw.csv')
        logger.info(f"[+] Loaded minority data: {len(minority_train)} train, {len(minority_test)} test")
    except FileNotFoundError:
        logger.error("[-] Minority data files not found. Run minority_wgan.py first.")
        return
    
    # Initialize preprocessor for info
    preprocessor = NSLKDDPreprocessor()
    
    # Show distributions before merge
    logger.info("[+] Majority train distribution:")
    logger.info(majority_train['Label'].value_counts())
    
    logger.info("[+] Minority train distribution:")
    logger.info(minority_train['Label'].value_counts())
    
    # Merge training data
    final_train = pd.concat([majority_train, minority_train], ignore_index=True)
    final_train = shuffle(final_train, random_state=42).reset_index(drop=True)
    
    # Merge test data
    final_test = pd.concat([majority_test, minority_test], ignore_index=True)
    final_test = shuffle(final_test, random_state=42).reset_index(drop=True)
    
    # Show final distributions
    logger.info("[+] Final training dataset:")
    preprocessor.info_dataset(final_train)
    
    logger.info("[+] Final test dataset:")
    preprocessor.info_dataset(final_test)
    
    # Save final datasets
    final_train.to_csv('data/final_train_balanced.csv', index=False)
    final_test.to_csv('data/final_test.csv', index=False)
    
    logger.info("[+] Final datasets saved:")
    logger.info(f"[+] - data/final_train_balanced.csv ({len(final_train)} samples)")
    logger.info(f"[+] - data/final_test.csv ({len(final_test)} samples)")
    
    # Create summary report
    create_summary_report(final_train, final_test)
    
    logger.info("[+] Final merge completed successfully!")

def create_summary_report(train_df, test_df):
    """
    Create a summary report of the final datasets
    """
    logger.info("[+] Creating summary report...")
    
    report = f"""
AWGAN Implementation - Final Dataset Summary
============================================

Training Dataset (Balanced):
- Total samples: {len(train_df)}
- Classes: {train_df['Label'].value_counts().to_dict()}

Test Dataset:
- Total samples: {len(test_df)}
- Classes: {test_df['Label'].value_counts().to_dict()}

Features:
- Total features: {len(train_df.columns) - 1}  # Excluding Label
- Categorical: protocol_type, service, flag
- Binary: land, logged_in, is_host_login, is_guest_login
- Numerical: {len(train_df.columns) - 7} features

Dataset Balance:
- Training: All classes have approximately 20,000 samples each
- Test: Original distribution preserved (no synthetic data)

Files Generated:
- data/final_train_balanced.csv: Balanced training dataset
- data/final_test.csv: Test dataset with original distribution
- data/majority_train_compressed_raw.csv: Compressed majority classes
- data/minority_train_augmented_raw.csv: Augmented minority classes
- models/wgan_*/: Trained WGAN models for each minority class
"""
    
    with open('data/dataset_summary.txt', 'w') as f:
        f.write(report)
    
    logger.info("[+] Summary report saved to data/dataset_summary.txt")

if __name__ == "__main__":
    setup_logging("INFO")
    merge_final_datasets() 