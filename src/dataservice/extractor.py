from typing import Dict, Set, List
import pandas as pd


class Extractor:
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