import pandas as pd
import numpy as np
from typing import Dict, Set, List
from sklearn.model_selection import train_test_split
from utils.logging import get_logger

logger = get_logger(__name__)

class DataService:
    """Static utility class for data processing operations"""

    @staticmethod
    def info_dataset(df: pd.DataFrame = None, label_column: str = None):
        logger.info(f"[+] Dataset shape: {df.shape}")
        # Label distribution - convert numpy array to pandas Series for value_counts
        if hasattr(df[label_column], 'value_counts'):
            logger.info(f"[+] Label distribution: {df[label_column].value_counts()}")
        else:
            # Convert numpy array to pandas Series
            label_series = pd.Series(df[label_column])
            logger.info(f"[+] Label distribution: {label_series.value_counts()}")
    
    @staticmethod
    def export_data(df: pd.DataFrame, file_path: str):
        """Export dataframe to CSV file"""
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
        """Split dataframe into train and test sets"""
        logger.debug(f"[+] Splitting data into train and test sets with test size {test_size}")
        return train_test_split(df, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def unique_values(csv_path: str, chunk_size: int = 500_000, columns: List[str] = None):
        # Track observed unique values for each column
        observed: Dict[str, Set] = {col: set() for col in columns}
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            for col in columns:
                vals = chunk[col].dropna().unique()
                if len(vals) == 0:
                    continue
                # Update observed set
                s = observed[col]
                for v in vals:
                    # Normalize strings: strip
                    if isinstance(v, str):
                        v = v.strip()
                    s.add(v)
        cols = {}
        for col, vals in observed.items():
            cols[col] = sorted(vals)
        return cols

