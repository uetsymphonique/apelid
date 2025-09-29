import os

ORIGINAL_DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/original"
DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
REPORT_FOLDER = "reports/cic2018"
ENCODERS_FOLDER = "encoders/cic2018"

# Folders of encoded datasets divided by label
# Naming convention: cic2018_<label>_encoded.csv (label = class_name.lower().replace(' ', '_').replace('/', '_'))
ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"

# Folders of raw processed datasets divided by label
# Naming convention: cic2018_<label>_raw_processed.csv (label = class_name.lower().replace(' ', '_').replace('/', '_'))
RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"

# Folder of clean merged datasets divided by label
# Naming convention: cic2018_<label>_clean_merged.csv (label = class_name.lower().replace(' ', '_').replace('/', '_'))
CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"


# Folder of compressed datasets divided by label
COMPRESSED_DATA_FOLDER = f"{DATA_FOLDER}/compressed"



EMBEDDINGS_FOLDER = f"{DATA_FOLDER}/embeddings"
PCA_CACHE_FOLDER = f"{EMBEDDINGS_FOLDER}/pca_cache"



LABEL_COLUMN = "Label"

MAJORITY_LABELS = [
    'Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
    'Infilteration', 'SSH-Bruteforce', 
]

MINORITY_LABELS = [
    'DoS attacks-GoldenEye',
    'DoS attacks-Slowloris',
    # 'DDOS attack-LOIC-UDP',
    'Brute Force -Web',
    'Brute Force -XSS',
    'SQL Injection',
    'DoS attacks-SlowHTTPTest',
    'FTP-BruteForce'
]


def get_label_name(class_name: str) -> str:
    return class_name.lower().replace(' ', '_').replace('/', '_')