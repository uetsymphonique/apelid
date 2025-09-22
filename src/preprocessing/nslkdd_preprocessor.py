from preprocessing.preprocessor import Preprocessor
import pandas as pd
from utils.logging import get_logger, setup_logging
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import joblib

logger = get_logger(__name__)

DOS_ATTACKS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
PROBE_ATTACKS = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
PRIVILEGE_ATTACKS = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                     'rootkit', 'sqlattack', 'xterm']
ACCESS_ATTACKS = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                  'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                  'xclock', 'xsnoop']

class NSLKDDPreprocessor(Preprocessor):
    def __init__(self):
        self.features = [
            'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login',
            'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate']
        self.label_column = 'Label'
        self.encoded_categorical_features = [
            'protocol_type', 'service', 'flag'
        ]
        self.encoded_numerical_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate'
        ]
        self.binary_features = ['land', 'logged_in', 'is_host_login', 'is_guest_login']
        self.encoders = {}
        self.encoders_dir = "encoders/nslkdd"


    def map_label(self, df: pd.DataFrame):
        def map_attack(attack):
            if attack in DOS_ATTACKS:
                return 'DoS'
            elif attack in PROBE_ATTACKS:
                return 'Probe'
            elif attack in PRIVILEGE_ATTACKS:
                return 'U2R'
            elif attack in ACCESS_ATTACKS:
                return 'R2L'
            else:
                return 'Benign'
        df['Label'] = df['attack'].apply(map_attack)
        return df
    
    def select_features_and_label(self, df: pd.DataFrame):
        return df[self.features + [self.label_column]]
        
    def setup_encoders(self, df: pd.DataFrame):
        self.encoders = {
            'categorical': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'numerical': MinMaxScaler(),
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder()
        }
        self.encoders['categorical'].fit(df[self.encoded_categorical_features])
        self.encoders['numerical'].fit(df[self.encoded_numerical_features])
        self.encoders['label'].fit(df[self.label_column])
        self.encoders['ordinal'].fit(df[self.encoded_categorical_features])
        

    def load_encoders(self, encoders_dir=None):
        """Load pre-trained encoders from saved files"""
        import joblib
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
        try:
            self.encoders = {
                'categorical': joblib.load(f"{encoders_dir}/categorical_encoder.pkl"),
                'numerical': joblib.load(f"{encoders_dir}/numerical_encoder.pkl"),
                'label': joblib.load(f"{encoders_dir}/label_encoder.pkl"),
                'ordinal': joblib.load(f"{encoders_dir}/ordinal_encoder.pkl")
            }
            logger.info(f"[+] Encoders loaded from {encoders_dir}")
            return True
        except FileNotFoundError as e:
            logger.error(f"[-] Encoder files not found in {encoders_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"[-] Error loading encoders: {e}")
            return False

    def save_encoders(self, encoders_dir=None):
        """Save trained encoders to files"""
        import joblib
        import os
        
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
            
        os.makedirs(encoders_dir, exist_ok=True)
        
        try:
            joblib.dump(self.encoders['categorical'], f"{encoders_dir}/categorical_encoder.pkl")
            joblib.dump(self.encoders['numerical'], f"{encoders_dir}/numerical_encoder.pkl")
            joblib.dump(self.encoders['label'], f"{encoders_dir}/label_encoder.pkl")
            joblib.dump(self.encoders['ordinal'], f"{encoders_dir}/ordinal_encoder.pkl")
            logger.info(f"[+] Encoders saved to {encoders_dir}")
            return True
        except Exception as e:
            logger.error(f"[-] Error saving encoders: {e}")
            return False

    def label_distribution(self, df: pd.DataFrame):
        return df[self.label_column].value_counts()
    
    def preprocess_encode_categorical_features(self, df: pd.DataFrame):
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
        scaler = self.encoders['numerical']
        df[self.encoded_numerical_features] = scaler.transform(df[self.encoded_numerical_features])
        return df

    def preprocess_encode_ordinal_features(self, df: pd.DataFrame):
        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features] = encoder.transform(df[self.encoded_categorical_features])
        df[self.encoded_categorical_features] = df[self.encoded_categorical_features].astype(int)
        return df
    
    def inverse_transform_ordinal_features(self, df: pd.DataFrame):
        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features] = encoder.inverse_transform(df[self.encoded_categorical_features])
        return df

    def preprocess_encode_binary_features(self, df: pd.DataFrame):
        """Encode binary features - keep them as 0/1"""
        # Binary features should remain as 0/1, no scaling needed
        for feature in self.binary_features:
            if feature in df.columns:
                # Ensure binary values (0 or 1)
                df[feature] = df[feature].astype(int)
        return df

    def preprocess_encode_label(self, df: pd.DataFrame):
        encoder = self.encoders['label']
        df[self.label_column] = encoder.transform(df[self.label_column])
        return df

    def inverse_transform(self, df: pd.DataFrame):
        """
        Inverse transform the encoded DataFrame back to original format
        """
            
        df_inverse = df.copy()
        
        # 1. Inverse transform categorical features (one-hot -> original strings)
        if 'categorical' in self.encoders:
            encoder = self.encoders['categorical']
            # Find one-hot columns
            onehot_columns = [col for col in df_inverse.columns 
                            if any(col.startswith(prefix) for prefix in ['protocol_type_', 'service_', 'flag_'])]
            
            if onehot_columns:
                # Extract one-hot data
                onehot_data = df_inverse[onehot_columns].values
                
                # Inverse transform to get original categorical values
                original_cat = encoder.inverse_transform(onehot_data)
                original_cat_df = pd.DataFrame(original_cat, 
                                             columns=['protocol_type', 'service', 'flag'],
                                             index=df_inverse.index)
                
                # Replace one-hot columns with original categorical columns
                df_inverse = df_inverse.drop(columns=onehot_columns)
                df_inverse = pd.concat([df_inverse, original_cat_df], axis=1)
                logger.debug(f"[+] Inverse transformed categorical features")
        
        # 2. Inverse transform numerical features (scaled -> original values)
        if 'numerical' in self.encoders:
            scaler = self.encoders['numerical']
            numerical_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_srv_rerror_rate']
            
            # Find which numerical features exist in current df
            existing_numerical = [col for col in numerical_features if col in df_inverse.columns]
            
            if existing_numerical:
                df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features")
        
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
        original_order = [
            'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login',
            'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', self.label_column
        ]
        
        # Only keep columns that exist
        existing_columns = [col for col in original_order if col in df_inverse.columns]
        df_inverse = df_inverse[existing_columns]
        
        logger.debug(f"[+] Inverse transform completed. Shape: {df_inverse.shape}")
        return df_inverse


