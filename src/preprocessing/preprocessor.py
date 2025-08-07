import pandas as pd
import numpy as np
from utils.logging import get_logger
import joblib

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self, features: list[str], label_column: str):
        self.features = features
        self.label_column = label_column
        self.encoders = {}

    
    def dump_encoders(self):
        for key, encoder in self.encoders.items():
            joblib.dump(encoder, f'models/{key}_encoder.pkl')
            logger.info(f"[+] Saved {key} encoder to models/{key}_encoder.pkl")

    def info_dataset(self, df: pd.DataFrame = None):
        logger.info(f"[+] Dataset shape: {df.shape}")
        # Label distribution - convert numpy array to pandas Series for value_counts
        if hasattr(df[self.label_column], 'value_counts'):
            logger.info(f"[+] Label distribution: {df[self.label_column].value_counts()}")
        else:
            # Convert numpy array to pandas Series
            label_series = pd.Series(df[self.label_column])
            logger.info(f"[+] Label distribution: {label_series.value_counts()}")
    