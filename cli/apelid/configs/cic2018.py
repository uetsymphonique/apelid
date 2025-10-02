from .resources import Resources


class CIC2018Resources(Resources):
    resources_name = "cic2018"
    ORIGINAL_DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/original"
    DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
    REPORT_FOLDER = "reports/cic2018"
    ENCODERS_FOLDER = "encoders/cic2018"

    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"
    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"

    EMBEDDINGS_FOLDER = f"{DATA_FOLDER}/embeddings"
    PCA_CACHE_FOLDER = f"{EMBEDDINGS_FOLDER}/pca_cache"

    LABEL_COLUMN = "Label"

    MAJORITY_LABELS = [
        'Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 
        'Infilteration', 'SSH-Bruteforce', 'DoS attacks-GoldenEye'
    ]

    MINORITY_LABELS = [
        'DoS attacks-Slowloris',
        # 'DDOS attack-LOIC-UDP',
        'Brute Force -Web',
        'Brute Force -XSS',
        'SQL Injection',
        # 'DoS attacks-SlowHTTPTest',
        # 'FTP-BruteForce'
    ]


# Backward compatibility module-level constants
ORIGINAL_DATA_FOLDER = CIC2018Resources.ORIGINAL_DATA_FOLDER
DATA_FOLDER = CIC2018Resources.DATA_FOLDER
REPORT_FOLDER = CIC2018Resources.REPORT_FOLDER
ENCODERS_FOLDER = CIC2018Resources.ENCODERS_FOLDER
ENCODED_DATA_FOLDER = CIC2018Resources.ENCODED_DATA_FOLDER
RAW_PROCESSED_DATA_FOLDER = CIC2018Resources.RAW_PROCESSED_DATA_FOLDER
CLEAN_MERGED_DATA_FOLDER = CIC2018Resources.CLEAN_MERGED_DATA_FOLDER
EMBEDDINGS_FOLDER = CIC2018Resources.EMBEDDINGS_FOLDER
PCA_CACHE_FOLDER = CIC2018Resources.PCA_CACHE_FOLDER
LABEL_COLUMN = CIC2018Resources.LABEL_COLUMN
MAJORITY_LABELS = CIC2018Resources.MAJORITY_LABELS
MINORITY_LABELS = CIC2018Resources.MINORITY_LABELS


def get_label_name(class_name: str) -> str:
    return CIC2018Resources.get_label_name(class_name)