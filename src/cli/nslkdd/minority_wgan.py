from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import get_logger, setup_logging
import pandas as pd
from resampling.data_augmentation.augmented_wgan.pipeline import generate_augmented_samples, AugmentOptions
from sklearn.model_selection import train_test_split
import torch

logger = get_logger(__name__)

def split_minority_classes(df, preprocessor, test_size=0.3, random_state=42):
    """
    Split minority classes into train and test sets
    According to AWGAN paper: Split FIRST, then augment train set
    """
    minority_classes = ['Probe', 'R2L', 'U2R']
    minority_dfs = []
    
    for class_name in minority_classes:
        logger.info(f"[+] Processing minority class: {class_name}")
        
        # Get data for this class
        class_data = df[df['Label'] == class_name].copy()
        logger.info(f"[+] {class_name} class has {len(class_data)} samples")
        
        if len(class_data) == 0:
            logger.warning(f"[-] No samples found for class {class_name}")
            continue
        
        # Split this class
        X_class = class_data.drop(columns=[preprocessor.label_column])
        y_class = class_data[preprocessor.label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class, test_size=test_size, random_state=random_state,
            stratify=y_class if y_class.nunique() > 1 else None
        )
        
        # Reconstruct DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        minority_dfs.append({
            'class_name': class_name,
            'train': train_df,
            'test': test_df
        })
        
        logger.info(f"[+] {class_name} split: {len(train_df)} train, {len(test_df)} test")
    
    return minority_dfs

def augment_minority_class(class_data, preprocessor, tau=14000, device='cpu'):
    """
    Augment minority class using WGAN to reach tau samples
    tau = 14000 (70% of 20000) to balance with compressed majority classes
    """
    class_name = class_data['class_name']
    train_df = class_data['train']
    test_df = class_data['test']
    
    # NSL-KDD specific accept rates based on class
    accept_rate_map = {'Probe': 0.3, 'R2L': 0.3, 'U2R': 0.3}
    ar = accept_rate_map.get(class_name, 0.3)
    
    # For NSL-KDD, we don't have benign encoded handy → use None benign loader
    def benign_loader():
        try:
            benign_encoded = pd.read_csv('data/majority_train_compressed_encoded.csv')
            benign_val = preprocessor.encoders['label'].transform(['Benign'])[0]
            return benign_encoded[benign_encoded['Label'] == benign_val]
        except Exception as e:
            logger.warning(f"[-] Could not load benign samples for critic: {e}")
            return None
    
    # Simplified options for NSL-KDD
    opts = AugmentOptions(
        use_benign_for_critic=True,
        critic_epochs=60,
        wgan_iterations=10000,
        d_iter=5,
        use_gp=True,
        accept_rate=ar,
        request_multiplier=2.0,  # Less aggressive for NSL-KDD
        max_rounds=20,           # Fewer rounds needed for NSL-KDD
        use_postfilter=False,    # Skip post-filter for NSL-KDD
        min_precision=0.9,
        use_encoded_dedup=False, # Skip encoded dedup for NSL-KDD
        use_raw_dedup=True,
        trim_to_need=True,
        use_final_fill=False,    # Skip final fill for NSL-KDD
    )
    
    return generate_augmented_samples(
        pre=preprocessor,
        class_name=class_name,
        train_df=train_df,
        test_df=test_df,
        benign_loader=benign_loader,
        tau=tau,
        device=device,
        accept_rate=ar,
        options=opts,
    )

def process_minority_classes(df, preprocessor, tau=14000, device='cpu'):
    """
    Process all minority classes: split, augment, and return results
    AWGAN Logic:
    1. Split minority classes 70/30 (train/test) FIRST
    2. Augment train set to tau = 14000 samples (70% of 20000)
    3. Keep test set unchanged
    This balances with majority classes which are compressed to 20000 then split 70/30
    """
    logger.info("[+] Processing minority classes...")
    
    # Split minority classes
    minority_splits = split_minority_classes(df, preprocessor)
    
    augmented_dfs = []
    test_dfs = []
    
    for class_data in minority_splits:
        class_name = class_data['class_name']
        
        # Augment training data
        augmented_train = augment_minority_class(class_data, preprocessor, tau, device)
        augmented_dfs.append(augmented_train)
        
        # Keep test data unchanged
        test_dfs.append(class_data['test'])
        
        logger.info(f"[+] Completed processing {class_name}")
    
    # Combine all augmented training data
    if augmented_dfs:
        augmented_majority = pd.concat(augmented_dfs, ignore_index=True)
        logger.info(f"[+] Total augmented minority samples: {len(augmented_majority)}")
    else:
        augmented_majority = pd.DataFrame()
        logger.warning("[-] No minority classes processed")
    
    # Combine all test data
    if test_dfs:
        minority_test = pd.concat(test_dfs, ignore_index=True)
        logger.info(f"[+] Total minority test samples: {len(minority_test)}")
    else:
        minority_test = pd.DataFrame()
    
    return augmented_majority, minority_test

if __name__ == "__main__":
    setup_logging("DEBUG")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[+] Using device: {device}")
    
    # Load minority classes data
    minority_df = pd.read_csv('data/minority_classes_raw.csv')
    logger.info(f"[+] Loaded minority classes: {len(minority_df)} samples")
    logger.info(f"[+] Minority label distribution:")
    logger.info(minority_df['Label'].value_counts())
    
    # Initialize preprocessor
    preprocessor = NSLKDDPreprocessor()
    
    # Try to load existing encoders first
    if not preprocessor.load_encoders():
        logger.info("[+] Loading encoders failed, setting up new encoders...")
        preprocessor.setup_encoders(minority_df)
        preprocessor.save_encoders()
    else:
        logger.info("[+] Successfully loaded existing encoders")
    
    # Process minority classes
    augmented_minority, minority_test = process_minority_classes(
        minority_df, preprocessor, tau=14000, device=device
    )
    
    # Save results
    if len(augmented_minority) > 0:
        preprocessor.info_dataset(augmented_minority)
        augmented_minority.to_csv('data/minority_train_augmented_raw.csv', index=False)
        logger.info("[+] Saved augmented minority training data")
    
    if len(minority_test) > 0:
        preprocessor.info_dataset(minority_test)
        minority_test.to_csv('data/minority_test_raw.csv', index=False)
        logger.info("[+] Saved minority test data")
    
    logger.info("[+] Minority WGAN processing completed!")
