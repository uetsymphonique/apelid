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


# ---------- Reusable path helpers ----------

def embedding_path(subset: str, label: str, *, filtered_benign: bool = False) -> str:
    """Return absolute path to embedding parquet for the given subset/label.

    - Benign may have a filtered variant stored with suffix _embedding_filtered.parquet
    - Others use base _embedding.parquet
    """
    label_safe = get_label_name(label)
    base_dir = os.path.join(EMBEDDINGS_FOLDER, subset)
    if filtered_benign and subset == 'train':
        return os.path.join(base_dir, f"cic2018_{label_safe}_embedding_filtered.parquet")
    return os.path.join(base_dir, f"cic2018_{label_safe}_embedding.parquet")


def kmeans_centers_path_train(label: str, *, benign_source: str | None = None) -> str:
    """Default path to store MiniBatchKMeans centers under embeddings/train.
    - Non-Benign: cic2018_<label>_kmeans_centers.npy
    - Benign: cic2018_benign_kmeans_centers_<source>.npy where <source> in {filtered, base}
    """
    label_safe = get_label_name(label)
    base_dir = os.path.join(EMBEDDINGS_FOLDER, 'train')
    if label == 'Benign' and benign_source in {'filtered', 'base'}:
        return os.path.join(base_dir, f"cic2018_{label_safe}_kmeans_centers_{benign_source}.npy")
    return os.path.join(base_dir, f"cic2018_{label_safe}_kmeans_centers.npy")