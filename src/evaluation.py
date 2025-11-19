"""
Evaluation module for unsupervised learning.
Implements clustering evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """Class for evaluating clustering results."""
    
    def __init__(self):
        """Initialize clustering evaluator."""
        self.metrics = {}
    
    def compute_silhouette_score(self, X, labels, metric='euclidean'):
        """
        Compute silhouette score for clustering.
        
        Args:
            X: Input data
            labels: Cluster labels
            metric: Distance metric
            
        Returns:
            Silhouette score
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Filter out noise points (-1 labels) for DBSCAN/HDBSCAN
        valid_mask = labels >= 0
        if not valid_mask.all():
            X_filtered = X_array[valid_mask]
            labels_filtered = labels[valid_mask]
            logger.info(f"Filtered {(~valid_mask).sum()} noise points for silhouette calculation")
        else:
            X_filtered = X_array
            labels_filtered = labels
        
        # Check if we have at least 2 clusters
        n_clusters = len(set(labels_filtered))
        if n_clusters < 2:
            logger.warning("Need at least 2 clusters for silhouette score")
            return None
        
        score = silhouette_score(X_filtered, labels_filtered, metric=metric)
        logger.info(f"Silhouette Score: {score:.4f}")
        
        return score
    
    def compute_davies_bouldin_score(self, X, labels):
        """
        Compute Davies-Bouldin index for clustering.
        Lower is better.
        
        Args:
            X: Input data
            labels: Cluster labels
            
        Returns:
            Davies-Bouldin index
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Filter out noise points
        valid_mask = labels >= 0
        if not valid_mask.all():
            X_filtered = X_array[valid_mask]
            labels_filtered = labels[valid_mask]
            logger.info(f"Filtered {(~valid_mask).sum()} noise points for Davies-Bouldin calculation")
        else:
            X_filtered = X_array
            labels_filtered = labels
        
        # Check if we have at least 2 clusters
        n_clusters = len(set(labels_filtered))
        if n_clusters < 2:
            logger.warning("Need at least 2 clusters for Davies-Bouldin index")
            return None
        
        score = davies_bouldin_score(X_filtered, labels_filtered)
        logger.info(f"Davies-Bouldin Index: {score:.4f}")
        
        return score
    
    def compute_calinski_harabasz_score(self, X, labels):
        """
        Compute Calinski-Harabasz index for clustering.
        Higher is better.
        
        Args:
            X: Input data
            labels: Cluster labels
            
        Returns:
            Calinski-Harabasz index
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Filter out noise points
        valid_mask = labels >= 0
        if not valid_mask.all():
            X_filtered = X_array[valid_mask]
            labels_filtered = labels[valid_mask]
            logger.info(f"Filtered {(~valid_mask).sum()} noise points for Calinski-Harabasz calculation")
        else:
            X_filtered = X_array
            labels_filtered = labels
        
        # Check if we have at least 2 clusters
        n_clusters = len(set(labels_filtered))
        if n_clusters < 2:
            logger.warning("Need at least 2 clusters for Calinski-Harabasz index")
            return None
        
        score = calinski_harabasz_score(X_filtered, labels_filtered)
        logger.info(f"Calinski-Harabasz Index: {score:.4f}")
        
        return score
    
    def evaluate_clustering(self, X, labels, method_name='clustering'):
        """
        Compute all clustering metrics.
        
        Args:
            X: Input data
            labels: Cluster labels
            method_name: Name of clustering method
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {method_name}")
        
        metrics = {
            'method': method_name,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_samples': len(labels),
            'n_noise': list(labels).count(-1) if -1 in labels else 0
        }
        
        # Compute metrics
        silhouette = self.compute_silhouette_score(X, labels)
        davies_bouldin = self.compute_davies_bouldin_score(X, labels)
        calinski_harabasz = self.compute_calinski_harabasz_score(X, labels)
        
        metrics['silhouette_score'] = silhouette
        metrics['davies_bouldin_index'] = davies_bouldin
        metrics['calinski_harabasz_index'] = calinski_harabasz
        
        # Store metrics
        self.metrics[method_name] = metrics
        
        return metrics
    
    def compare_methods(self, results_dict):
        """
        Compare multiple clustering methods.
        
        Args:
            results_dict: Dictionary with {method_name: metrics_dict}
            
        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(results_dict).T
        
        logger.info("Clustering methods comparison:")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
    
    def plot_silhouette_analysis(self, X, labels, method_name='clustering', save_path=None):
        """
        Create silhouette analysis plot.
        
        Args:
            X: Input data
            labels: Cluster labels
            method_name: Name of clustering method
            save_path: Path to save plot
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Filter out noise points
        valid_mask = labels >= 0
        X_filtered = X_array[valid_mask]
        labels_filtered = labels[valid_mask]
        
        n_clusters = len(set(labels_filtered))
        
        if n_clusters < 2:
            logger.warning("Need at least 2 clusters for silhouette analysis")
            return
        
        # Compute silhouette scores
        silhouette_avg = silhouette_score(X_filtered, labels_filtered)
        sample_silhouette_values = silhouette_samples(X_filtered, labels_filtered)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[labels_filtered == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title(f"Silhouette Analysis for {method_name}\n(Average Score: {silhouette_avg:.3f})")
        
        # Vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
        
        ax.set_yticks([])
        ax.set_xlim([-0.1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved silhouette analysis plot to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, comparison_df, save_path=None):
        """
        Plot comparison of metrics across methods.
        
        Args:
            comparison_df: DataFrame with metrics comparison
            save_path: Path to save plot
        """
        metrics_to_plot = ['silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_index']
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("No metrics available to plot")
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Plot bars
            comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Method')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison plot to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module ready for use.")
