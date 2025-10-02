from .resources import Resources


class NSLKDDResources(Resources):
    resources_name = "nslkdd"
    DATA_FOLDER = "/dis/DS/minhtq/NSLKDD"

    KDD_TEXT_PATH = f"{DATA_FOLDER}/KDD+.txt"
    NSLKDD_ORIGINAL_CSV_PATH = f"{DATA_FOLDER}/nslkdd_original.csv"

    CLEAN_MERGED_DATA_FOLDER = f"{DATA_FOLDER}/clean_merged"
    ENCODED_DATA_FOLDER = f"{DATA_FOLDER}/encoded"
    RAW_PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/raw_processed"

    ENCODERS_FOLDER = "encoders/nslkdd"
    REPORT_FOLDER = "reports/nslkdd"

    MAJORITY_LABELS = ['Benign', 'DoS']
    MINORITY_LABELS = ['Probe', 'R2L', 'U2R']


# Backward compatibility module-level constants
DATA_FOLDER = NSLKDDResources.DATA_FOLDER
KDD_TEXT_PATH = NSLKDDResources.KDD_TEXT_PATH
NSLKDD_ORIGINAL_CSV_PATH = NSLKDDResources.NSLKDD_ORIGINAL_CSV_PATH
CLEAN_MERGED_DATA_FOLDER = NSLKDDResources.CLEAN_MERGED_DATA_FOLDER
ENCODED_DATA_FOLDER = NSLKDDResources.ENCODED_DATA_FOLDER
RAW_PROCESSED_DATA_FOLDER = NSLKDDResources.RAW_PROCESSED_DATA_FOLDER
ENCODERS_FOLDER = NSLKDDResources.ENCODERS_FOLDER
REPORT_FOLDER = NSLKDDResources.REPORT_FOLDER
MAJORITY_LABELS = NSLKDDResources.MAJORITY_LABELS
MINORITY_LABELS = NSLKDDResources.MINORITY_LABELS


def get_label_name(class_name: str) -> str:
    return NSLKDDResources.get_label_name(class_name)