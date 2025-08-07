from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import get_logger, setup_logging
import pandas as pd
from datasvc.data_service import DataService
from models.wgan import WGAN
from sklearn.model_selection import train_test_split
import numpy as np
import os
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
    
    logger.info(f"[+] Augmenting {class_name} from {len(train_df)} to {tau} samples")
    
    # Encode the training data
    train_encoded = preprocessor.preprocess_encode_numerical_features(train_df.copy())
    train_encoded = preprocessor.preprocess_encode_binary_features(train_encoded)
    train_encoded = preprocessor.preprocess_encode_label(train_encoded)
    train_encoded = preprocessor.preprocess_encode_categorical_features(train_encoded)
    
    logger.info(f"[+] Encoded training data shape: {train_encoded.shape}")
    
    # Initialize WGAN
    x_dim = train_encoded.shape[1] - 1  # Exclude Label column
    # Use WGAN-GP for better stability if needed
    use_gradient_penalty = True  # Set to False for original WGAN
    wgan = WGAN(x_dim=x_dim, device=device, use_gp=use_gradient_penalty,
                use_critic_loss=True, lambda_critic=0.5)
    
    if use_gradient_penalty:
        logger.info("[+] Using WGAN-GP (Gradient Penalty) for improved stability")
    else:
        logger.info("[+] Using original WGAN with weight clipping")
    
    # -------------------------------------------------
    # Prepare data for Critic training: add benign
    # -------------------------------------------------
    try:
        benign_encoded = pd.read_csv('data/majority_train_compressed_encoded.csv')
        benign_val = preprocessor.encoders['label'].transform(['Benign'])[0]
        benign_encoded = benign_encoded[benign_encoded['Label'] == benign_val]
        benign_sample = benign_encoded.sample(n=len(train_encoded)*4, replace=True, random_state=1)
        critic_df = pd.concat([train_encoded, benign_sample], ignore_index=True).sample(frac=1, random_state=1)
        logger.info(f"[+] Critic training set: {critic_df['Label'].value_counts().to_dict()}")
    except Exception as e:
        logger.warning(f"[-] Could not load benign samples for critic: {e}. Training critic on attack-only data")
        critic_df = train_encoded.copy()

    # Train critic on combined data (use_label_column=True)
    critic_loader, _ = wgan.prepare_data(critic_df, use_label_column=True)
    wgan.train_critic(critic_loader, epochs=60)

    # Prepare data loader for WGAN training (attack samples only)
    attack_loader, _ = wgan.prepare_data(train_encoded, use_label_column=False)
    
    # Train WGAN
    wgan.train_wgan(attack_loader, iterations=10000, d_iter=5, save_interval=1000)
    
    # Generate samples to reach tau
    current_samples = len(train_df)
    samples_needed = tau - current_samples
    
    if samples_needed > 0:
        logger.info(f"[+] Generating {samples_needed} additional samples...")
        # Threshold per class for critic
        threshold_map = {'Probe': 0.85, 'R2L': 0.6, 'U2R': 0.75}
        thr = threshold_map.get(class_name, 0.8)
        logger.info(f"[+] Using critic_threshold={thr} for class {class_name}")
        generated_samples = wgan.generate_samples(samples_needed, critic_threshold=thr)
        
        # Convert generated samples to DataFrame
        feature_names = [col for col in train_encoded.columns if col != 'Label']
        generated_df = pd.DataFrame(generated_samples, columns=feature_names)
        
        # Add label column with original encoded value for this attack class
        # Get the encoded value for this class
        class_encoded_value = preprocessor.encoders['label'].transform([class_name])[0]
        generated_df['Label'] = class_encoded_value
        
        # Inverse transform to get original format
        generated_original = preprocessor.inverse_transform(generated_df)
        
        # Combine with original training data
        augmented_train = pd.concat([train_df, generated_original], ignore_index=True)
        
        logger.info(f"[+] {class_name} augmented: {len(train_df)} → {len(augmented_train)} samples")
        
        # Save models for this class
        model_dir = f"models/wgan_{class_name.lower()}"
        wgan.save_models(model_dir)
        wgan.plot_losses(f"wgan_losses_{class_name.lower()}.png")
        
        return augmented_train
    else:
        logger.info(f"[+] {class_name} already has enough samples ({current_samples} >= {tau})")
        return train_df

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
