import pandas as pd
from preprocessing.nslkdd_preprocessor import NSLKDDPreprocessor
from dataservice.data_service import DataService
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


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
    
    
    # Fix duplicates and missing values
    original_df_svc.fix_duplicates()
    original_df_svc.fix_missing_values()
    logger.info(f"[+] Original dataset shape: {original_df_svc.df.shape}")
    logger.info(f"[+] Original label distribution:")
    logger.info(original_df_svc.df['Label'].value_counts())
