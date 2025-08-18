from preprocessing.preprocessor import Preprocessor
import pandas as pd
import numpy as np
import os
from utils.logging import get_logger, setup_logging
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import joblib

logger = get_logger(__name__)

class CIC2018Preprocessor(Preprocessor):
    def __init__(self):
        
        # Original features
        # 'Dst Port', 'Protocol', # 2
        # 'Timestamp', # dropped(1) # 3
        # 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', # 7
        # 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', # 11
        # 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', # 15
        # 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 18
        # 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 22
        # 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 27
        # 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 32
        # 'Fwd PSH Flags', # 33
        # 'Bwd PSH Flags', # dropped(2) # 34
        # 'Fwd URG Flags', # 35
        # 'Bwd URG Flags', # dropped(3) # 36
        # 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', # 40
        # 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', # 45
        # 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', # 50
        # 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', # 53
        # 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 57
        # 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', # dropped(9) # 63
        # 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 67
        # 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 71
        # 'Active Mean', 'Active Std', 'Active Max', 'Active Min', # 75
        # 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', # 79
        # 'Label' # 80
        
        # Features to keep after dropping constants and timestamp
        self.features = [
            'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', # 5
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', # 9
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', # 13
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 17
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 21
            'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 26
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 31
            'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', # 35
            'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', # 40
            'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', # 45
            'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', # 50
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 54
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 58
            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 62
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', # 68
            'Idle Max', 'Idle Min' # 70
        ]
        
        self.label_column = 'Label'

        self.cat_features = ['Protocol', 'Dst Port']
        self.cont_features =  [
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', # 3
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', # 7
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', # 11
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 15
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 19
            'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 24
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 29
            'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', # 33
            'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', # 38
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 42
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 46
            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 50
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', # 56
            'Idle Max', 'Idle Min' # 58
        ]

        # Binary features (keep as 0/1, no scaling)
        self.binary_features = [
            'Fwd PSH Flags', 'Fwd URG Flags', 'FIN Flag Cnt', 'SYN Flag Cnt', # 4
            'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', # 8
            'CWE Flag Count', 'ECE Flag Cnt' # 10
        ]

        ### in theory, Protocol and Dst Port are categorical features, but Dst Port has a very large range of values (1..65535)
        ### so we will treat it as numerical for now but it will be encoded as ordinal when used for training with boosting models and DNN
        ### Protocol and Dst Port will be ignored when we do approximately deduplication (approx_dedup_features)
        
        # Categorical features for OneHot encoding
        self.encoded_categorical_features = ['Protocol'] # 1

        # Ordinal-encoded features (for classical models); include 'Dst Port'
        # For 'Dst Port', we will fit full range [1..65535] to handle unseen values
        self.encoded_categorical_features_ordinal = self.cat_features
        
        
        # Numerical features for MinMax scaling (all features minus categorical and binary)
        self.encoded_numerical_features = [
            'Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', # 4
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', # 8
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', # 12
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', # 16
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', # 20
            'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', # 25
            'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', # 30
            'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', # 34
            'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', # 39
            'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', # 43
            'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', # 47
            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', # 51
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', # 57
            'Idle Max', 'Idle Min' # 59
        ]

        self.numerical_features_for_approx_dedup = self.cont_features
        
        self.encoders = {
            'categorical': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'numerical': MinMaxScaler(),
            'label': LabelEncoder(),
            # OrdinalEncoder will be instantiated with explicit categories below
        }

    def select_features_and_label(self, df: pd.DataFrame):
        """Select features and label, dropping unwanted columns"""
        # Drop unwanted columns if they exist
        columns_to_drop = [
            'Timestamp',  # Drop timestamp to avoid overfitting
            'Bwd PSH Flags', 'Bwd URG Flags',  # Constant columns
            'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',  # Constant columns
            'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'   # Constant columns
        ]
        
        available_features = [col for col in self.features if col in df.columns]
        available_drops = [col for col in columns_to_drop if col in df.columns]
        
        if available_drops:
            logger.info(f"[+] Dropping columns: {available_drops}")
            df = df.drop(columns=available_drops)
        
        selected_columns = available_features + [self.label_column]
        return df[selected_columns]
        
    def setup_encoders(self, df: pd.DataFrame):
        """Setup encoders for CIC-2018 features"""
        
        # Handle infinity values before fitting encoders
        df_clean = df.copy()
        
        # Replace infinity with NaN for numerical features
        if self.encoded_numerical_features and all(col in df_clean.columns for col in self.encoded_numerical_features):
            
            df_clean[self.encoded_numerical_features] = df_clean[self.encoded_numerical_features].replace([np.inf, -np.inf], np.nan)
            # Drop rows with NaN in numerical features
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=self.encoded_numerical_features)
            after_count = len(df_clean)
            if before_count != after_count:
                logger.info(f"[+] Dropped {before_count - after_count} rows with infinity/NaN in numerical features")
        
        # Fit encoders
        if self.encoded_categorical_features and all(col in df_clean.columns for col in self.encoded_categorical_features):
            self.encoders['categorical'].fit(df_clean[self.encoded_categorical_features])

        # Prepare OrdinalEncoder with categories (Protocol from data, Dst Port full range)
        if self.encoded_categorical_features_ordinal and all(col in df_clean.columns for col in self.encoded_categorical_features_ordinal):
            categories = []
            for col in self.encoded_categorical_features_ordinal:
                if col == 'Dst Port':
                    categories.append(np.arange(1, 65536, dtype=int))
                else:
                    cats = pd.Series(df_clean[col]).dropna().unique().tolist()
                    categories.append(sorted(cats))
            self.encoders['ordinal'] = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
            self.encoders['ordinal'].fit(df_clean[self.encoded_categorical_features_ordinal])
        
        if self.encoded_numerical_features and all(col in df_clean.columns for col in self.encoded_numerical_features):
            self.encoders['numerical'].fit(df_clean[self.encoded_numerical_features])
        
        if self.label_column in df_clean.columns:
            self.encoders['label'].fit(df_clean[self.label_column])
        
        logger.info(f"[+] Encoders fitted for CIC-2018 dataset")
        logger.info(f"[+] Categorical features: {self.encoded_categorical_features}")
        logger.info(f"[+] Binary features: {self.binary_features}")
        logger.info(f"[+] Numerical features: {len(self.encoded_numerical_features)} features")

    def load_encoders(self, encoders_dir="encoders"):
        """Load pre-trained encoders from saved files"""
        try:
            self.encoders = {
                'categorical': joblib.load(f"{encoders_dir}/cic_categorical_encoder.pkl"),
                'numerical': joblib.load(f"{encoders_dir}/cic_numerical_encoder.pkl"),
                'label': joblib.load(f"{encoders_dir}/cic_label_encoder.pkl"),
                'ordinal': joblib.load(f"{encoders_dir}/cic_ordinal_encoder.pkl")
            }
            logger.info(f"[+] CIC-2018 encoders loaded from {encoders_dir}")
            return True
        except FileNotFoundError as e:
            logger.error(f"[-] CIC-2018 encoder files not found in {encoders_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"[-] Error loading CIC-2018 encoders: {e}")
            return False

    def save_encoders(self, encoders_dir="encoders"):
        """Save trained encoders to files"""
        
        os.makedirs(encoders_dir, exist_ok=True)
        
        try:
            joblib.dump(self.encoders['categorical'], f"{encoders_dir}/cic_categorical_encoder.pkl")
            joblib.dump(self.encoders['numerical'], f"{encoders_dir}/cic_numerical_encoder.pkl")
            joblib.dump(self.encoders['label'], f"{encoders_dir}/cic_label_encoder.pkl")
            joblib.dump(self.encoders['ordinal'], f"{encoders_dir}/cic_ordinal_encoder.pkl")
            logger.info(f"[+] CIC-2018 encoders saved to {encoders_dir}")
            return True
        except Exception as e:
            logger.error(f"[-] Error saving CIC-2018 encoders: {e}")
            return False

    def label_distribution(self, df: pd.DataFrame):
        return df[self.label_column].value_counts()
    
    def preprocess_encode_categorical_features(self, df: pd.DataFrame):
        """Encode categorical features using OneHot encoding"""
        if not self.encoded_categorical_features or not all(col in df.columns for col in self.encoded_categorical_features):
            return df
            
        encoder = self.encoders['categorical']
        encoded_train = encoder.transform(df[self.encoded_categorical_features])
        encoded_train_df = pd.DataFrame(
            encoded_train, 
            columns=encoder.get_feature_names_out(self.encoded_categorical_features),
            index=df.index
        )
        df = pd.concat([encoded_train_df, df], axis=1)
        df = df.drop(columns=self.encoded_categorical_features)
        return df

    def preprocess_encode_numerical_features(self, df: pd.DataFrame):
        """Encode numerical features using MinMax scaling"""
        if not self.encoded_numerical_features or not all(col in df.columns for col in self.encoded_numerical_features):
            return df
            
        # Handle infinity values before scaling
        df_clean = df.copy()
        df_clean[self.encoded_numerical_features] = df_clean[self.encoded_numerical_features].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN in numerical features
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=self.encoded_numerical_features)
        after_count = len(df_clean)
        if before_count != after_count:
            logger.info(f"[+] Dropped {before_count - after_count} rows with infinity/NaN during encoding")
            
        scaler = self.encoders['numerical']
        df_clean[self.encoded_numerical_features] = scaler.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_ordinal_features(self, df: pd.DataFrame):
        """Encode categorical features using Ordinal encoding"""
        if not self.encoded_categorical_features_ordinal or not all(col in df.columns for col in self.encoded_categorical_features_ordinal):
            return df
            
        # Sanitize 'Dst Port' to avoid unknown mapping (-1)
        if 'Dst Port' in df.columns:
            # Coerce to numeric
            df['Dst Port'] = pd.to_numeric(df['Dst Port'], errors='coerce')
            before_na = df['Dst Port'].isna().sum()
            # Round and clip to valid range, fill NaN to 1
            df['Dst Port'] = (
                df['Dst Port']
                .round()
                .clip(lower=1, upper=65535)
                .fillna(1)
                .astype(int)
            )
            if before_na:
                logger.debug(f"[+] Coerced and filled {before_na} NaN 'Dst Port' values before ordinal encoding")

        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features_ordinal] = encoder.transform(df[self.encoded_categorical_features_ordinal])
        df[self.encoded_categorical_features_ordinal] = df[self.encoded_categorical_features_ordinal].astype(int)
        return df
    
    def inverse_transform_ordinal_features(self, df: pd.DataFrame):
        """Inverse transform ordinal features"""
        if not self.encoded_categorical_features_ordinal or not all(col in df.columns for col in self.encoded_categorical_features_ordinal):
            return df
            
        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features_ordinal] = encoder.inverse_transform(df[self.encoded_categorical_features_ordinal])
        return df

    def preprocess_encode_binary_features(self, df: pd.DataFrame):
        """Encode binary features - keep them as 0/1"""
        for feature in self.binary_features:
            if feature in df.columns:
                # Ensure binary values (0 or 1)
                df[feature] = df[feature].astype(int)
        return df

    def preprocess_encode_label(self, df: pd.DataFrame):
        """Encode labels using LabelEncoder"""
        if self.label_column not in df.columns:
            return df
            
        encoder = self.encoders['label']
        df[self.label_column] = encoder.transform(df[self.label_column])
        return df

    def inverse_transform(self, df: pd.DataFrame):
        """
        Inverse transform the encoded DataFrame back to original format
        """
        df_inverse = df.copy()
        
        # 1. Inverse transform categorical features (one-hot -> original strings)
        if 'categorical' in self.encoders and self.encoded_categorical_features:
            encoder = self.encoders['categorical']
            # Find one-hot columns for Protocol
            onehot_columns = [col for col in df_inverse.columns if col.startswith('Protocol_')]
            
            if onehot_columns:
                # Extract one-hot data
                onehot_data = df_inverse[onehot_columns].values
                
                # Inverse transform to get original categorical values
                original_cat = encoder.inverse_transform(onehot_data)
                original_cat_df = pd.DataFrame(original_cat, 
                                             columns=['Protocol'],
                                             index=df_inverse.index)
                
                # Replace one-hot columns with original categorical columns
                df_inverse = df_inverse.drop(columns=onehot_columns)
                df_inverse = pd.concat([df_inverse, original_cat_df], axis=1)
                logger.debug(f"[+] Inverse transformed categorical features")
        
        # 2. Inverse transform numerical features (scaled -> original values)
        if 'numerical' in self.encoders and self.encoded_numerical_features:
            scaler = self.encoders['numerical']
            # Find which numerical features exist in current df
            existing_numerical = [col for col in self.encoded_numerical_features if col in df_inverse.columns]
            
            if existing_numerical:
                df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features")

        # 2b. Fix 'Dst Port' to be an integer in [1, 65535]
        if 'Dst Port' in df_inverse.columns:
            try:
                df_inverse['Dst Port'] = (
                    df_inverse['Dst Port']
                    .round()
                    .clip(lower=1, upper=65535)
                    .astype(int)
                )
                logger.debug("[+] Normalized 'Dst Port' to integer range [1, 65535]")
            except Exception as e:
                logger.warning(f"[!] Could not normalize 'Dst Port': {e}")
        
        # 3. Inverse transform labels (encoded -> original strings)
        if 'label' in self.encoders and self.label_column in df_inverse.columns:
            label_encoder = self.encoders['label']
            df_inverse[self.label_column] = label_encoder.inverse_transform(df_inverse[self.label_column])
            logger.debug(f"[+] Inverse transformed labels")
        
        # 4. Fix binary features - convert back to 0/1
        for feature in self.binary_features:
            if feature in df_inverse.columns:
                # Convert to binary using threshold 0.5
                df_inverse[feature] = (df_inverse[feature] > 0.5).astype(int)
                logger.debug(f"[+] Fixed binary feature {feature}: converted to 0/1")
        
        # 5. Reorder columns to match original order
        original_order = self.features + [self.label_column]
        
        # Only keep columns that exist
        existing_columns = [col for col in original_order if col in df_inverse.columns]
        df_inverse = df_inverse[existing_columns]
        
        logger.debug(f"[+] Inverse transform completed. Shape: {df_inverse.shape}")
        return df_inverse

    def export_encoded_data(self, df: pd.DataFrame, file_path: str):
        """
        Export encoded DataFrame to CSV file
        """
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"[+] Encoded data exported to: {file_path}")
            logger.info(f"[+] Shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"[-] Error exporting encoded data to {file_path}: {e}")
            return False

    def export_raw_data(self, df: pd.DataFrame, file_path: str):
        """
        Export raw DataFrame (after inverse transform) to CSV file
        """
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"[+] Raw data exported to: {file_path}")
            logger.info(f"[+] Shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"[-] Error exporting raw data to {file_path}: {e}")
            return False
            

    

