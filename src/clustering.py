"""
Clustering module for unsupervised learning.
Implements KMeans, DBSCAN, and other clustering algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Class for clustering analysis."""
    
    def __init__(self):
        """Initialize clustering analyzer."""
        self.kmeans_model = None
        self.dbscan_model = None
        self.hdbscan_model = None
        self.hierarchical_model = None
        self.labels = {}
        self.cluster_centers = {}
    
    def apply_kmeans(self, X, n_clusters=3, random_state=42, **kwargs):
        """
        Apply KMeans clustering.
        
        Args:
            X: Input data
            n_clusters: Number of clusters
            random_state: Random seed
            **kwargs: Additional KMeans parameters
            
        Returns:
            Cluster labels and model
        """
        logger.info(f"Applying KMeans with {n_clusters} clusters")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit KMeans
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            **kwargs
        )
        
        labels = self.kmeans_model.fit_predict(X_array)
        
        logger.info(f"KMeans inertia: {self.kmeans_model.inertia_:.2f}")
        logger.info(f"Cluster sizes: {np.bincount(labels)}")
        
        # Store results
        self.labels['kmeans'] = labels
        self.cluster_centers['kmeans'] = self.kmeans_model.cluster_centers_
        
        return labels, self.kmeans_model
    
    def apply_dbscan(self, X, eps=0.5, min_samples=5, metric='euclidean', **kwargs):
        """
        Apply DBSCAN clustering.
        
        Args:
            X: Input data
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            metric: Distance metric
            **kwargs: Additional DBSCAN parameters
            
        Returns:
            Cluster labels and model
        """
        logger.info(f"Applying DBSCAN with eps={eps}, min_samples={min_samples}")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit DBSCAN
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            **kwargs
        )
        
        labels = self.dbscan_model.fit_predict(X_array)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN found {n_clusters} clusters")
        logger.info(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
        
        if n_clusters > 0:
            cluster_sizes = np.bincount(labels[labels >= 0])
            logger.info(f"Cluster sizes: {cluster_sizes}")
        
        # Store results
        self.labels['dbscan'] = labels
        
        return labels, self.dbscan_model
    
    def apply_hdbscan(self, X, min_cluster_size=5, min_samples=None, **kwargs):
        """
        Apply HDBSCAN clustering.
        
        Args:
            X: Input data
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples parameter
            **kwargs: Additional HDBSCAN parameters
            
        Returns:
            Cluster labels and model
        """
        logger.info(f"Applying HDBSCAN with min_cluster_size={min_cluster_size}")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit HDBSCAN
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            **kwargs
        )
        
        labels = self.hdbscan_model.fit_predict(X_array)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"HDBSCAN found {n_clusters} clusters")
        logger.info(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
        
        # Store results
        self.labels['hdbscan'] = labels
        
        return labels, self.hdbscan_model
    
    def apply_hierarchical(self, X, n_clusters=3, linkage='ward', **kwargs):
        """
        Apply Hierarchical clustering.
        
        Args:
            X: Input data
            n_clusters: Number of clusters
            linkage: Linkage method ('ward', 'complete', 'average')
            **kwargs: Additional parameters
            
        Returns:
            Cluster labels and model
        """
        logger.info(f"Applying Hierarchical clustering with {n_clusters} clusters")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit Hierarchical
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        labels = self.hierarchical_model.fit_predict(X_array)
        
        logger.info(f"Cluster sizes: {np.bincount(labels)}")
        
        # Store results
        self.labels['hierarchical'] = labels
        
        return labels, self.hierarchical_model
    
    def elbow_method(self, X, max_clusters=10, save_path=None):
        """
        Apply elbow method to find optimal number of clusters.
        
        Args:
            X: Input data
            max_clusters: Maximum number of clusters to try
            save_path: Path to save plot
            
        Returns:
            Dictionary with inertias for each k
        """
        logger.info(f"Running elbow method for k=1 to {max_clusters}")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_array)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(alpha=0.3)
        plt.xticks(k_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved elbow plot to {save_path}")
        
        plt.show()
        
        return dict(zip(k_range, inertias))
    
    def plot_cluster_distribution(self, method='kmeans', save_path=None):
        """
        Plot distribution of cluster sizes.
        
        Args:
            method: Clustering method name
            save_path: Path to save plot
        """
        if method not in self.labels:
            logger.error(f"Method {method} not found in labels")
            return
        
        labels = self.labels[method]
        
        # Handle noise points for DBSCAN/HDBSCAN
        if -1 in labels:
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            cluster_names = ['Noise' if x == -1 else f'Cluster {x}' for x in cluster_counts.index]
        else:
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            cluster_names = [f'Cluster {x}' for x in cluster_counts.index]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(cluster_counts)), cluster_counts.values)
        
        # Color noise points differently if present
        if -1 in labels:
            bars[0].set_color('red')
        
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        plt.title(f'Cluster Size Distribution ({method.upper()})')
        plt.xticks(range(len(cluster_counts)), cluster_names, rotation=45)
        plt.grid(alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster distribution plot to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_scatter(self, X_2d, method='kmeans', title=None, save_path=None):
        """
        Plot clusters in 2D space.
        
        Args:
            X_2d: 2D data for plotting
            method: Clustering method name
            title: Plot title
            save_path: Path to save plot
        """
        if method not in self.labels:
            logger.error(f"Method {method} not found in labels")
            return
        
        labels = self.labels[method]
        
        plt.figure(figsize=(10, 8))
        
        # Handle noise points for DBSCAN/HDBSCAN
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                color = 'black'
                marker = 'x'
                alpha = 0.3
            else:
                marker = 'o'
                alpha = 0.6
            
            mask = labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[color], label=f'Cluster {label}' if label >= 0 else 'Noise',
                       alpha=alpha, s=50, marker=marker)
        
        # Plot cluster centers for KMeans
        if method == 'kmeans' and method in self.cluster_centers:
            centers_2d = self.cluster_centers[method][:, :2]  # Take first 2 dimensions
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='*', s=500, edgecolors='black', 
                       linewidths=2, label='Centroids', zorder=10)
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title or f'Cluster Visualization ({method.upper()})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster scatter plot to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_profile(self, X, method='kmeans', feature_names=None):
        """
        Get statistical profile of each cluster.
        
        Args:
            X: Original data
            method: Clustering method name
            feature_names: List of feature names
            
        Returns:
            DataFrame with cluster profiles
        """
        if method not in self.labels:
            logger.error(f"Method {method} not found in labels")
            return None
        
        labels = self.labels[method]
        
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feature_names = X.columns.tolist()
        else:
            df = pd.DataFrame(X, columns=feature_names or [f'Feature_{i}' for i in range(X.shape[1])])
        
        df['cluster'] = labels
        
        # Calculate cluster profiles
        profiles = df.groupby('cluster').agg(['mean', 'std', 'min', 'max', 'count'])
        
        return profiles


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Clustering module ready for use.")
