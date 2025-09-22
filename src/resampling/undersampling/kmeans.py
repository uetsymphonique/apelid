import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from utils.logging import get_logger
import time
from tqdm import tqdm

logger = get_logger(__name__)

class KMeansCompressor:
    def __init__(self, tau=20000, random_state=42):
        self.tau = tau
        self.random_state = random_state

    def compress_majority_class(self, X_class, y_class, n_clusters: int | None = None):
        start_time = time.time()
        # logger.info(f"[+] Starting compression for class with {len(X_class)} samples")
        
        if len(X_class) <= self.tau:
            logger.info(f"[+] Class size ({len(X_class)}) <= tau ({self.tau}), no compression needed")
            return X_class, y_class
        
        if n_clusters is None:
            # Use fewer clusters to ensure we get enough samples
            if self.tau * 3 < len(X_class):
                n_clusters = self.tau
            else:
                n_clusters = min(self.tau // 2, len(X_class) // 2)  # Use half of tau as clusters
            logger.info(f"[+] Using {n_clusters} clusters for {len(X_class)} samples")

            if n_clusters > 45000:
                logger.warning(f"[-] n_clusters ({n_clusters}) is greater than 45000, setting to 45000")
                n_clusters = 45000
        
        if n_clusters < 2:
            logger.warning(f"[-] Only {n_clusters} clusters found, using random sampling instead")
            return X_class.sample(n=self.tau, random_state=self.random_state), y_class.sample(n=self.tau, random_state=self.random_state)
            
        # KMeans clustering with progress tracking
        logger.debug(f"[+] Starting K-Means clustering...")
        kmeans_start = time.time()
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            batch_size=1000,
            verbose=0,
            max_iter=100
        )
        
        # Fit KMeans before accessing cluster_centers_
        kmeans.fit(X_class)
        
        kmeans_time = time.time() - kmeans_start
        logger.debug(f"[+] K-Means completed in {kmeans_time:.2f} seconds")
        
        # Choose samples from each cluster with progress tracking
        logger.debug(f"[+] Selecting representative samples from {n_clusters} clusters...")
        selection_start = time.time()
        
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        selected_indices = []
        
        # Progress bar for cluster processing
        with tqdm(total=n_clusters, desc="Processing Clusters", unit="cluster") as pbar:
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0:
                    pbar.set_postfix({"Cluster": i, "Status": "Empty"})
                    pbar.update(1)
                    continue
                    
                # Calculate how many samples to take from this cluster
                samples_per_cluster = max(1, len(cluster_indices) * self.tau // len(X_class))
                samples_per_cluster = min(samples_per_cluster, len(cluster_indices))
                
                # Choose samples closest to centroid
                cluster_points = X_class.iloc[cluster_indices]
                dists = np.linalg.norm(cluster_points.values - centroids[i], axis=1)
                
                # Sort by distance and take top samples_per_cluster
                sorted_indices = np.argsort(dists)
                selected_from_cluster = cluster_indices[sorted_indices[:samples_per_cluster]]
                selected_indices.extend(selected_from_cluster)
                
                pbar.set_postfix({
                    "Cluster": i, 
                    "Size": len(cluster_indices), 
                    "Selected": samples_per_cluster,
                    "Total": len(selected_indices)
                })
                pbar.update(1)
        
        logger.debug(f"[+] Selected {len(selected_indices)} samples from clusters")
        
        # If we don't have enough samples, add more randomly
        if len(selected_indices) < self.tau:
            remaining_indices = list(set(range(len(X_class))) - set(selected_indices))
            additional_needed = self.tau - len(selected_indices)
            logger.debug(f"[+] Need {additional_needed} more samples, adding randomly...")
            
            if len(remaining_indices) >= additional_needed:
                additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                selected_indices.extend(additional_indices)
        
        # Take exactly tau samples
        selected_indices = selected_indices[:self.tau]
        selection_time = time.time() - selection_start
        logger.debug(f"[+] Sample selection completed in {selection_time:.2f} seconds")
        
        X_selected = X_class.iloc[selected_indices]
        y_selected = y_class.iloc[selected_indices]
        
        total_time = time.time() - start_time
        logger.debug(f"[+] Compression completed: {len(X_class)} → {len(X_selected)} samples in {total_time:.2f}s")
        logger.debug(f"[+] Time breakdown: K-Means: {kmeans_time:.2f}s, Selection: {selection_time:.2f}s")
        
        return X_selected, y_selected 