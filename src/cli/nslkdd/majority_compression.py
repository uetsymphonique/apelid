from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from utils.logging import get_logger, setup_logging
import pandas as pd
from dataservice.data_service import DataService
from resampling.undersampling.kmeans import KMeansCompressor
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

def compress_majority_classes(df, preprocessor, tau=20000):
    """
    Compress majority classes (Benign, DoS) to tau samples each
    AWGAN Logic:
    1. Compress majority classes to tau = 20000 samples each
    2. Then split 70/30 (train/test)
    3. Result: ~14000 train, ~6000 test per class
    """
    compressor = KMeansCompressor(tau=tau, use_enn=False)
    compressed_dfs = []
    
    # Process each majority class
    majority_classes = preprocessor.encoders['label'].transform(['Benign', 'DoS'])
    
    for class_name in majority_classes:
        logger.info(f"[+] Processing majority class: {class_name}")
        
        # Get data for this class
        class_data = df[df['Label'] == class_name].copy()
        logger.info(f"[+] {class_name} class has {len(class_data)} samples")
        
        if len(class_data) <= tau:
            logger.info(f"[+] {class_name} class size ({len(class_data)}) <= tau ({tau}), no compression needed")
            compressed_dfs.append(class_data)
            continue
        
        # Separate features and labels
        X_class = class_data.drop(columns=[preprocessor.label_column])
        y_class = class_data[preprocessor.label_column]
        
        # Compress using KMeans + ENN
        X_compressed, y_compressed = compressor.compress_majority_class(X_class, y_class)
        
        # Reconstruct DataFrame
        compressed_df = pd.concat([X_compressed, y_compressed], axis=1)
        compressed_dfs.append(compressed_df)
        
        logger.info(f"[+] {class_name} compressed from {len(class_data)} to {len(compressed_df)} samples")
    
    # Combine all compressed majority classes
    compressed_majority = pd.concat(compressed_dfs, ignore_index=True)
    logger.info(f"[+] Total compressed majority samples: {len(compressed_majority)}")
    
    return compressed_majority

def split_train_test(df, test_size=0.3, random_state=42):
    """
    Split data into train and test sets
    """
    logger.debug(f"[+] Splitting data into train ({1-test_size:.1%}) and test ({test_size:.1%}) sets")
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Reconstruct DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, test_df

if __name__ == "__main__":
    setup_logging("INFO")
    # Load data
    nslkdd_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
    ]
    
    df = pd.read_csv(r'data/KDD+.txt', names=nslkdd_columns)
    preprocessor = NSLKDDPreprocessor()
    
    # Step 1: Map labels and select features
    df = preprocessor.map_label(df)
    df = preprocessor.select_features_and_label(df)
    
    # Step 2: Data cleaning
    original_df_svc = DataService(df)
    logger.info(f"[+] Original dataset shape: {original_df_svc.df.shape}")
    logger.info(f"[+] Original label distribution:")
    logger.info(original_df_svc.df['Label'].value_counts())
    
    # Fix duplicates and missing values
    original_df_svc.fix_duplicates()
    original_df_svc.fix_missing_values()
    
    # Step 3: Setup encoders
    preprocessor.setup_encoders(original_df_svc.df)
    preprocessor.save_encoders()
    
    # Step 4: Separate majority and minority classes
    majority_classes = ['Benign', 'DoS']
    minority_classes = ['Probe', 'R2L', 'U2R']
    
    majority_df = original_df_svc.df[original_df_svc.df['Label'].isin(majority_classes)].copy()
    minority_df = original_df_svc.df[original_df_svc.df['Label'].isin(minority_classes)].copy()
    
    logger.info(f"[+] Majority classes: {len(majority_df)} samples")
    logger.info(f"[+] Minority classes: {len(minority_df)} samples")

    # export majority and minority classes to csv
    majority_df.to_csv('data/majority_classes_raw.csv', index=False)
    minority_df.to_csv('data/minority_classes_raw.csv', index=False)
    
    # Step 5: Encode majority classes
    logger.info("[+] Encoding majority classes...")
    majority_df = preprocessor.preprocess_encode_numerical_features(majority_df)
    majority_df = preprocessor.preprocess_encode_binary_features(majority_df)
    majority_df = preprocessor.preprocess_encode_label(majority_df)
    majority_df = preprocessor.preprocess_encode_categorical_features(majority_df)
    
    logger.info(f"[+] Encoded majority data shape: {majority_df.shape}")
    logger.info(f"[+] Encoded majority label distribution:")
    logger.info(majority_df['Label'].value_counts())
    
    # Step 6: Compress majority classes
    tau_major = 22000
    compressed_majority = compress_majority_classes(majority_df, preprocessor, tau=tau_major)

    # ---------- ENN global refinement & adjust to 20k ----------
    from resampling.undersampling.enn_refiner import ENNRefiner
    refiner = ENNRefiner(tau_final=20000)
    compressed_majority = refiner.refine(compressed_majority, label_col=preprocessor.label_column)

    preprocessor.info_dataset(compressed_majority)

    majority_train, majority_test = split_train_test(compressed_majority, test_size=0.3, random_state=42)

    # preprocessor.info_dataset(majority_train)
    majority_train.to_csv('data/majority_train_compressed_encoded.csv', index=False)
    # preprocessor.info_dataset(majority_test)
    majority_test.to_csv('data/majority_test_compressed_encoded.csv', index=False)

    # inverse transform for train and test set
    majority_train = preprocessor.inverse_transform(majority_train)
    majority_test = preprocessor.inverse_transform(majority_test)

    # preprocessor.info_dataset(majority_train)
    majority_train.to_csv('data/majority_train_compressed_raw.csv', index=False)
    # preprocessor.info_dataset(majority_test)
    majority_test.to_csv('data/majority_test_compressed_raw.csv', index=False)