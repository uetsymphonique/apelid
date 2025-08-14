import os
import pandas as pd
from preprocessing.cic2018_preprocessor import CIC2018Preprocessor

DATA_FOLDER = "/dis/DS/minhtq/CIC-2018/"
RAW_PATH = os.path.join(DATA_FOLDER, "CIC2018_raw_processed.csv")


def main():
    df = pd.read_csv(RAW_PATH)
    pre = CIC2018Preprocessor()
    pre.select_features_and_label(df)
    pre.setup_encoders(df)
    pre.save_encoders()


if __name__ == "__main__":
    main()