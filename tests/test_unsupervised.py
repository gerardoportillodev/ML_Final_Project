"""Unit tests for unsupervised learning modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import DataLoader
from src.preprocessing import DataPreprocessor
from src.dim_reduction import DimensionalityReducer
from src.clustering import ClusteringAnalyzer
from src.evaluation import ClusteringEvaluator


class TestPreprocessing:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100) * 10,
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'feature4': np.random.randn(100)
        })
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        # Introduce missing values
        data = sample_data.copy()
        data.loc[0:5, 'feature1'] = np.nan
        
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.handle_missing_values(data)
        
        assert cleaned.isnull().sum().sum() == 0
    
    def test_encode_categorical(self, sample_data):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor()
        encoded = preprocessor.encode_categorical(sample_data.copy())
        
        # Should have more columns after one-hot encoding
        assert encoded.shape[1] >= sample_data.shape[1]


class TestDimensionalityReduction:
    """Tests for DimensionalityReducer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(np.random.rand(100, 10))
    
    def test_pca_application(self, sample_data):
        """Test PCA application."""
        reducer = DimensionalityReducer()
        X_reduced, model = reducer.apply_pca(sample_data, n_components=3)
        
        assert X_reduced.shape[1] == 3
        assert model is not None
    
    def test_umap_application(self, sample_data):
        """Test UMAP application."""
        reducer = DimensionalityReducer()
        X_reduced, model = reducer.apply_umap(sample_data, n_components=2)
        
        assert X_reduced.shape[1] == 2
        assert model is not None


class TestClustering:
    """Tests for ClusteringAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.rand(100, 5)
    
    def test_kmeans_clustering(self, sample_data):
        """Test KMeans clustering."""
        clusterer = ClusteringAnalyzer()
        labels, model = clusterer.apply_kmeans(sample_data, n_clusters=3)
        
        assert len(labels) == 100
        assert len(set(labels)) == 3
        assert model is not None
    
    def test_dbscan_clustering(self, sample_data):
        """Test DBSCAN clustering."""
        clusterer = ClusteringAnalyzer()
        labels, model = clusterer.apply_dbscan(sample_data, eps=0.5, min_samples=5)
        
        assert len(labels) == 100
        assert model is not None


class TestEvaluation:
    """Tests for ClusteringEvaluator class."""
    
    @pytest.fixture
    def sample_data_and_labels(self):
        """Create sample data and labels for testing."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        labels = np.random.choice([0, 1, 2], 100)
        return X, labels
    
    def test_silhouette_score(self, sample_data_and_labels):
        """Test silhouette score computation."""
        X, labels = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        score = evaluator.compute_silhouette_score(X, labels)
        
        assert score is not None
        assert -1 <= score <= 1
    
    def test_davies_bouldin_score(self, sample_data_and_labels):
        """Test Davies-Bouldin index computation."""
        X, labels = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        score = evaluator.compute_davies_bouldin_score(X, labels)
        
        assert score is not None
        assert score >= 0
    
    def test_calinski_harabasz_score(self, sample_data_and_labels):
        """Test Calinski-Harabasz index computation."""
        X, labels = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        score = evaluator.compute_calinski_harabasz_score(X, labels)
        
        assert score is not None
        assert score >= 0
    
    def test_evaluate_clustering(self, sample_data_and_labels):
        """Test complete clustering evaluation."""
        X, labels = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        metrics = evaluator.evaluate_clustering(X, labels, 'test')
        
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'calinski_harabasz_index' in metrics


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
