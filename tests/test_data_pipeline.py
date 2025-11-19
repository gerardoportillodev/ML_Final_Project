"""Unit tests for the data pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.config_loader import load_config


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': ['A', 'B', 'A', 'B', 'A'],
            'default': [0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def temp_csv(self, sample_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data is None
        assert loader.config is not None
    
    def test_load_data(self, temp_csv):
        """Test data loading."""
        loader = DataLoader()
        data = loader.load_data(temp_csv)
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == 5
        assert 'feature1' in data.columns
    
    def test_get_data_info(self, temp_csv):
        """Test getting data information."""
        loader = DataLoader()
        loader.load_data(temp_csv)
        info = loader.get_data_info()
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert info['shape'] == (5, 4)
    
    def test_clean_data_drop_duplicates(self):
        """Test cleaning data with duplicate removal."""
        loader = DataLoader()
        data_with_duplicates = pd.DataFrame({
            'feature1': [1, 2, 2, 3],
            'feature2': [10, 20, 20, 30],
            'default': [0, 1, 1, 0]
        })
        loader.data = data_with_duplicates
        
        cleaned = loader.clean_data(drop_duplicates=True, handle_missing='drop')
        assert cleaned.shape[0] == 3  # One duplicate removed
    
    def test_clean_data_handle_missing(self):
        """Test cleaning data with missing value handling."""
        loader = DataLoader()
        data_with_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [10, 20, 30, 40],
            'default': [0, 1, 0, 1]
        })
        loader.data = data_with_missing
        
        cleaned = loader.clean_data(drop_duplicates=False, handle_missing='drop')
        assert cleaned.shape[0] == 3  # One row with missing value removed


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'num_feature1': [1, 2, 3, 4, 5],
            'num_feature2': [10, 20, 30, 40, 50],
            'cat_feature': ['A', 'B', 'A', 'B', 'A'],
            'default': [0, 1, 0, 1, 0]
        })
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.scaler is not None
        assert engineer.feature_names == []
    
    def test_identify_features(self, sample_data):
        """Test feature identification."""
        engineer = FeatureEngineer()
        numerical, categorical = engineer.identify_features(
            sample_data, target_column='default'
        )
        
        assert len(numerical) == 2
        assert 'num_feature1' in numerical
        assert 'num_feature2' in numerical
        assert len(categorical) == 1
        assert 'cat_feature' in categorical
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical feature encoding."""
        engineer = FeatureEngineer()
        encoded = engineer.encode_categorical_features(
            sample_data, ['cat_feature'], fit=True
        )
        
        assert encoded['cat_feature'].dtype in [np.int32, np.int64]
        assert 'cat_feature' in engineer.label_encoders
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        engineer = FeatureEngineer()
        scaled = engineer.scale_features(
            sample_data, ['num_feature1', 'num_feature2'], fit=True
        )
        
        # Check that features are scaled (mean ~0, std ~1)
        assert abs(scaled['num_feature1'].mean()) < 1e-10
        assert abs(scaled['num_feature2'].mean()) < 1e-10
    
    def test_prepare_features(self, sample_data):
        """Test complete feature preparation."""
        engineer = FeatureEngineer()
        X, y = engineer.prepare_features(sample_data, target_column='default', fit=True)
        
        assert X.shape[0] == 5
        assert y.shape[0] == 5
        assert 'default' not in X.columns
        assert len(engineer.feature_names) > 0
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        engineer = FeatureEngineer()
        X, y = engineer.prepare_features(sample_data, target_column='default', fit=True)
        
        X_train, X_test, y_train, y_test = engineer.split_data(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 4
        assert X_test.shape[0] == 1
        assert y_train.shape[0] == 4
        assert y_test.shape[0] == 1


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(18, 70, 100),
            'income': np.random.randint(20000, 100000, 100),
            'credit_score': np.random.randint(300, 850, 100),
            'employment': np.random.choice(['employed', 'unemployed'], 100),
            'default': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
    
    @pytest.fixture
    def temp_csv(self, sample_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_complete_pipeline(self, temp_csv):
        """Test complete data processing pipeline."""
        # Load data
        loader = DataLoader()
        data = loader.load_data(temp_csv)
        
        # Get data info
        info = loader.get_data_info()
        assert info['shape'][0] > 0
        
        # Clean data
        cleaned_data = loader.clean_data()
        
        # Prepare features
        engineer = FeatureEngineer()
        X, y = engineer.prepare_features(cleaned_data, target_column='default', fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = engineer.split_data(X, y, test_size=0.2)
        
        # Verify pipeline output
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
        assert len(engineer.feature_names) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
