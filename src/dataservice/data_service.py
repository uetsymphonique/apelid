import pandas as pd
import numpy as np
from typing import Dict, Set, List
from sklearn.model_selection import train_test_split
from utils.logging import get_logger

logger = get_logger(__name__)

class DataService:
    """Static utility class for data processing operations"""
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> bool:
        """Check if dataframe contains duplicates"""
        return df.duplicated().any()
    
    @staticmethod
    def check_missing_and_inf_values(df: pd.DataFrame) -> bool:
        """Check if the dataframe contains any missing values (NaN) or infinite values (+inf or -inf)"""
        has_missing = df.isnull().any().any()
        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        return has_missing or has_inf
    
    @staticmethod
    def fix_duplicates(df: pd.DataFrame, method="auto") -> pd.DataFrame:
        """
        Remove duplicates using memory-efficient methods for large datasets.
        
        Args:
            df: DataFrame to deduplicate
            method: "auto", "pandas", "streaming", or "external"
                - auto: Choose method based on dataset size
                - pandas: Standard pandas drop_duplicates (fast but memory-intensive)
                - streaming: Chunk-based processing (memory-efficient)
                - external: Use external sorting (most memory-efficient)
        
        Returns:
            Deduplicated DataFrame
        """
        original_count = len(df)
        dataset_size_gb = df.memory_usage(deep=True).sum() / 1024**3
        
        logger.debug(f"[+] Dropping duplicates from {original_count} rows ({dataset_size_gb:.1f}GB)")
        
        # Auto-select method based on dataset size
        if method == "auto":
            if dataset_size_gb < 2.0:
                method = "pandas"
            elif dataset_size_gb < 8.0:
                method = "streaming"
            else:
                method = "external"
        
        logger.debug(f"[+] Using {method} method for deduplication")
        if method == "pandas":
            result_df = DataService._fix_duplicates_pandas(df)
        elif method == "streaming":
            result_df = DataService._fix_duplicates_streaming(df)
        elif method == "external":
            result_df = DataService._fix_duplicates_external(df)
        else:
            raise ValueError(f"Unknown deduplication method: {method}")
        
        final_count = len(result_df)
        duplicates_removed = original_count - final_count
        logger.debug(f"[+] Method: {method}, Removed {duplicates_removed} duplicates, Final: {final_count} rows")
        
        return result_df
    
    @staticmethod
    def _fix_duplicates_pandas(df: pd.DataFrame) -> pd.DataFrame:
        """Standard pandas deduplication - fast but memory-intensive"""
        return df.drop_duplicates()
    
    @staticmethod
    def _fix_duplicates_streaming(df: pd.DataFrame, chunksize=100_000) -> pd.DataFrame:
        """Memory-efficient streaming deduplication using Bloom filter + external sort"""
        import tempfile
        import os
        import hashlib
        
        logger.debug(f"[+] Using streaming deduplication with chunksize={chunksize}")
        
        # Create temporary files
        temp_chunks_dir = tempfile.mkdtemp(prefix='dedup_chunks_')
        temp_final = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False)
        temp_final_path = temp_final.name
        temp_final.close()
        
        try:
            # Use simple bloom filter approach with limited memory
            max_seen_hashes = 10_000_000  # Limit to ~80MB for hash tracking
            seen_hashes = set()
            chunk_files = []
            total_processed = 0
            
            logger.debug(f"[+] Processing {len(df)} rows in chunks of {chunksize}")
            
            # Phase 1: Process chunks and write unique-within-chunk to separate files
            for chunk_idx, start_idx in enumerate(range(0, len(df), chunksize)):
                end_idx = min(start_idx + chunksize, len(df))
                chunk = df.iloc[start_idx:end_idx].copy()
                
                # Remove duplicates within this chunk first (fast)
                chunk_unique = chunk.drop_duplicates()
                
                # Create stable hash for cross-chunk deduplication
                chunk_hashes = []
                for _, row in chunk_unique.iterrows():
                    # Create deterministic hash from row values
                    row_str = '|'.join(str(v) for v in row.values)
                    hash_val = int(hashlib.md5(row_str.encode()).hexdigest()[:16], 16)
                    chunk_hashes.append(hash_val)
                
                # Filter against seen hashes (if we haven't exceeded limit)
                if len(seen_hashes) < max_seen_hashes:
                    new_mask = [h not in seen_hashes for h in chunk_hashes]
                    chunk_filtered = chunk_unique[new_mask]
                    
                    # Update seen hashes
                    seen_hashes.update(h for i, h in enumerate(chunk_hashes) if new_mask[i])
                else:
                    # If hash set is full, keep all (will sort later)
                    chunk_filtered = chunk_unique
                    logger.debug(f"[+] Hash limit exceeded, will rely on external sort for final dedup")
                
                # Write chunk to temporary file
                if len(chunk_filtered) > 0:
                    chunk_file = os.path.join(temp_chunks_dir, f'chunk_{chunk_idx:06d}.csv')
                    chunk_filtered.to_csv(chunk_file, index=False)
                    chunk_files.append(chunk_file)
                    total_processed += len(chunk_filtered)
                
                # Memory cleanup
                del chunk, chunk_unique, chunk_hashes, chunk_filtered
                
                if chunk_idx % 10 == 0:
                    logger.debug(f"[+] Processed chunk {chunk_idx+1}, {total_processed} rows so far")
            
            logger.debug(f"[+] Phase 1 complete: {len(chunk_files)} chunk files, {total_processed} rows")
            
            # Phase 2: Merge chunk files using external sort for final deduplication
            if len(chunk_files) == 0:
                # Empty result
                result_df = pd.DataFrame(columns=df.columns)
            elif len(chunk_files) == 1:
                # Single chunk, just read it
                result_df = pd.read_csv(chunk_files[0], low_memory=False)
            else:
                # Multiple chunks, use external sort for final merge + dedup
                logger.debug(f"[+] Phase 2: External sort merge of {len(chunk_files)} files")
                
                # Get header from first file
                header_df = pd.read_csv(chunk_files[0], nrows=0)
                header_line = ','.join(header_df.columns)
                
                # Write header to final file
                with open(temp_final_path, 'w') as f:
                    f.write(header_line + '\n')
                
                # Concatenate all chunk files and sort
                import subprocess
                cat_cmd = f"cat {' '.join(chunk_files)} | tail -n +2 | sort -T /tmp -u >> '{temp_final_path}'"
                result = subprocess.run(cat_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"External merge failed: {result.stderr}")
                
                # Read final result
                result_df = pd.read_csv(temp_final_path, low_memory=False)
            
            logger.debug(f"[+] Streaming dedup completed: {len(result_df)} unique rows")
            return result_df
            
        finally:
            # Cleanup temp files
            import shutil
            if os.path.exists(temp_chunks_dir):
                shutil.rmtree(temp_chunks_dir)
            if os.path.exists(temp_final_path):
                os.remove(temp_final_path)
    
    @staticmethod
    def _fix_duplicates_external(df: pd.DataFrame) -> pd.DataFrame:
        """External sorting-based deduplication - most memory efficient"""
        import tempfile
        import subprocess
        import os
        
        logger.debug("[+] Using external sorting deduplication")
        
        # Create temporary files
        temp_input = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False)
        temp_output = tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False)
        temp_input_path = temp_input.name
        temp_output_path = temp_output.name
        temp_input.close()
        temp_output.close()
        
        try:
            # Save current dataframe to temp file
            df.to_csv(temp_input_path, index=False)
            
            # Get header
            header = df.columns.tolist()
            header_line = ','.join(header)
            
            # Use external sort to deduplicate
            # First write header
            with open(temp_output_path, 'w') as f:
                f.write(header_line + '\n')
            
            # Sort and deduplicate data lines (skip header)
            sort_cmd = f"tail -n +2 '{temp_input_path}' | sort -T /tmp -u >> '{temp_output_path}'"
            result = subprocess.run(sort_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"External sort failed: {result.stderr}")
            
            # Read back deduplicated data
            result_df = pd.read_csv(temp_output_path, low_memory=False)
            logger.debug("[+] External dedup completed")
            return result_df
            
        finally:
            # Cleanup temp files
            for path in [temp_input_path, temp_output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    @staticmethod
    def fix_missing_and_inf_values(df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN and drop all rows containing NaN or infinite values"""
        logger.debug(f"[+] Cleaning missing and infinite values from {df.shape[0]} rows")
        # Replace +inf and -inf with NaN
        result_df = df.replace([np.inf, -np.inf], np.nan)
        # Drop rows containing NaN (including original NaN and converted inf values)
        result_df = result_df.dropna()
        logger.debug(f"[+] to {result_df.shape[0]} rows")
        return result_df
    
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

