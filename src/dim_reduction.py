"""
Dimensionality reduction module for unsupervised learning.
Implements PCA, UMAP, and other reduction techniques.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Class for dimensionality reduction methods."""
    
    def __init__(self):
        """Initialize dimensionality reducer."""
        self.pca_model = None
        self.umap_model = None
        self.tsne_model = None
        self.reduced_data = {}
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Input data (DataFrame or array)
            n_components: Number of components (if None, uses variance_threshold)
            variance_threshold: Cumulative variance to retain
            
        Returns:
            Reduced data and PCA model
        """
        logger.info("Applying PCA")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns
        else:
            X_array = X
            feature_names = None
        
        # Fit PCA
        if n_components is None:
            self.pca_model = PCA(n_components=variance_threshold, svd_solver='full')
        else:
            self.pca_model = PCA(n_components=n_components)
        
        X_reduced = self.pca_model.fit_transform(X_array)
        
        # Log results
        n_components_actual = self.pca_model.n_components_
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info(f"PCA reduced from {X_array.shape[1]} to {n_components_actual} components")
        logger.info(f"Explained variance: {explained_variance[:5]}")
        logger.info(f"Cumulative variance: {cumulative_variance[:5]}")
        
        # Store results
        self.reduced_data['pca'] = X_reduced
        
        return X_reduced, self.pca_model
    
    def apply_umap(self, X, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        """
        Apply UMAP for dimensionality reduction.
        
        Args:
            X: Input data
            n_components: Number of dimensions for output
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            metric: Distance metric
            
        Returns:
            Reduced data and UMAP model
        """
        logger.info(f"Applying UMAP with {n_components} components")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit UMAP
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        
        X_reduced = self.umap_model.fit_transform(X_array)
        
        logger.info(f"UMAP reduced from {X_array.shape[1]} to {n_components} components")
        
        # Store results
        self.reduced_data['umap'] = X_reduced
        
        return X_reduced, self.umap_model
    
    def apply_tsne(self, X, n_components=2, perplexity=30, n_iter=1000):
        """
        Apply t-SNE for dimensionality reduction.
        
        Args:
            X: Input data
            n_components: Number of dimensions for output
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            
        Returns:
            Reduced data and t-SNE model
        """
        logger.info(f"Applying t-SNE with {n_components} components")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Fit t-SNE
        self.tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42
        )
        
        X_reduced = self.tsne_model.fit_transform(X_array)
        
        logger.info(f"t-SNE reduced from {X_array.shape[1]} to {n_components} components")
        
        # Store results
        self.reduced_data['tsne'] = X_reduced
        
        return X_reduced, self.tsne_model
    
    def plot_explained_variance(self, save_path=None):
        """
        Plot explained variance for PCA.
        
        Args:
            save_path: Path to save plot
        """
        if self.pca_model is None:
            logger.warning("PCA model not fitted yet")
            return
        
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(alpha=0.3)
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved explained variance plot to {save_path}")
        
        plt.show()
    
    def plot_2d_projection(self, method='pca', labels=None, title=None, save_path=None):
        """
        Plot 2D projection of reduced data.
        
        Args:
            method: 'pca', 'umap', or 'tsne'
            labels: Cluster labels for coloring
            title: Plot title
            save_path: Path to save plot
        """
        if method not in self.reduced_data:
            logger.error(f"Method {method} not found in reduced data")
            return
        
        X_reduced = self.reduced_data[method]
        
        if X_reduced.shape[1] < 2:
            logger.error(f"Need at least 2 components for 2D plot, got {X_reduced.shape[1]}")
            return
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=50)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(title or f'2D {method.upper()} Projection')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D projection to {save_path}")
        
        plt.show()
    
    def plot_3d_projection(self, method='pca', labels=None, title=None, save_path=None):
        """
        Plot 3D projection of reduced data.
        
        Args:
            method: 'pca', 'umap', or 'tsne'
            labels: Cluster labels for coloring
            title: Plot title
            save_path: Path to save plot
        """
        if method not in self.reduced_data:
            logger.error(f"Method {method} not found in reduced data")
            return
        
        X_reduced = self.reduced_data[method]
        
        if X_reduced.shape[1] < 3:
            logger.error(f"Need at least 3 components for 3D plot, got {X_reduced.shape[1]}")
            return
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                               c=labels, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
        ax.set_title(title or f'3D {method.upper()} Projection')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D projection to {save_path}")
        
        plt.show()
    
    def get_pca_loadings(self, feature_names, n_components=5):
        """
        Get PCA loadings (feature contributions).
        
        Args:
            feature_names: List of feature names
            n_components: Number of components to show
            
        Returns:
            DataFrame with loadings
        """
        if self.pca_model is None:
            logger.error("PCA model not fitted yet")
            return None
        
        loadings = pd.DataFrame(
            self.pca_model.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
        
        return loadings


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Dimensionality reduction module ready for use.")
