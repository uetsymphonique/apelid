import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logging import get_logger

logger = get_logger(__name__)

class DataService:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def update_df(self, df: pd.DataFrame):
        self.df = df

    def check_duplicates(self):
        return self.df.duplicated().any()
    
    def check_missing_values(self):
        return self.df.isnull().any().any()
    
    def fix_duplicates(self):
        logger.debug(f"[+] Dropping duplicates from {self.df.shape[0]} rows")
        self.df = self.df.drop_duplicates()
        logger.debug(f"[+] to {self.df.shape[0]} rows")
    
    def fix_missing_values(self):
        self.df.dropna()
    
    def export_data(self, file_path: str):
        self.df.to_csv(file_path, index=False)
    
    def split_data(self, test_size: float = 0.3, random_state: int = 42):
        logger.debug(f"[+] Splitting data into train and test sets with test size {test_size}")
        return train_test_split(self.df, test_size=test_size, random_state=random_state)

