ORIGINAL_DATA_FOLDER = "/dis/DS/CIC2018/"
DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"

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


CLEAN_MERGED_FILE_NAME = "CIC2018_clean_merged.csv"
RAW_PROCESSED_FILE_NAME = "CIC2018_raw_processed.csv"
ENCODED_FILE_NAME = "CIC2018_encoded.csv"

def get_label_name(class_name: str) -> str:
    return class_name.lower().replace(' ', '_').replace('/', '_')