"""Feature engineering module for Credit Risk ML project."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from src.logger import setup_logger
from src.config_loader import load_config, get_absolute_path


logger = setup_logger(__name__)


class FeatureEngineer:
    """Class for feature engineering and transformation."""
    
    def __init__(self, config=None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config if config else load_config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def identify_features(self, data, target_column=None):
        """
        Identify numerical and categorical features.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        if target_column is None:
            target_column = self.config['features'].get('target_column', 'default')
        
        # Exclude target column
        feature_columns = [col for col in data.columns if col != target_column]
        
        # Identify numerical and categorical features
        numerical_features = data[feature_columns].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        categorical_features = data[feature_columns].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        logger.info(f"Identified {len(numerical_features)} numerical features")
        logger.info(f"Identified {len(categorical_features)} categorical features")
        
        return numerical_features, categorical_features
    
    def encode_categorical_features(self, data, categorical_features, fit=True):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            data: Input DataFrame
            categorical_features: List of categorical feature names
            fit: Whether to fit the encoders (True for training, False for test)
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        data = data.copy()
        
        for feature in categorical_features:
            if fit:
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(
                    data[feature].astype(str)
                )
                logger.info(f"Encoded categorical feature: {feature}")
            else:
                if feature in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[feature]
                    data[feature] = data[feature].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    logger.info(f"Applied encoding to feature: {feature}")
        
        return data
    
    def scale_features(self, data, numerical_features, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            data: Input DataFrame
            numerical_features: List of numerical feature names
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        data = data.copy()
        
        if fit:
            data[numerical_features] = self.scaler.fit_transform(
                data[numerical_features]
            )
            logger.info(f"Scaled {len(numerical_features)} numerical features")
        else:
            data[numerical_features] = self.scaler.transform(
                data[numerical_features]
            )
            logger.info(f"Applied scaling to {len(numerical_features)} features")
        
        return data
    
    def create_interaction_features(self, data, feature_pairs=None):
        """
        Create interaction features from pairs of numerical features.
        
        Args:
            data: Input DataFrame
            feature_pairs: List of tuples with feature pairs to interact
            
        Returns:
            pd.DataFrame: DataFrame with interaction features added
        """
        if feature_pairs is None:
            logger.info("No feature pairs specified for interaction")
            return data
        
        data = data.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in data.columns and feat2 in data.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                data[interaction_name] = data[feat1] * data[feat2]
                logger.info(f"Created interaction feature: {interaction_name}")
        
        return data
    
    def prepare_features(self, data, target_column=None, fit=True):
        """
        Prepare features for model training.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            fit: Whether to fit transformers (True for training, False for test)
            
        Returns:
            tuple: (X, y) - features and target
        """
        if target_column is None:
            target_column = self.config['features'].get('target_column', 'default')
        
        logger.info("Starting feature preparation")
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            logger.warning(f"Target column '{target_column}' not found in data")
            X = data
            y = None
        
        # Identify feature types
        numerical_features, categorical_features = self.identify_features(
            data, target_column
        )
        
        # Encode categorical features
        if categorical_features:
            X = self.encode_categorical_features(X, categorical_features, fit=fit)
        
        # Scale numerical features
        if numerical_features:
            X = self.scale_features(X, numerical_features, fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
            logger.info(f"Stored {len(self.feature_names)} feature names")
        
        logger.info(f"Feature preparation completed. X shape: {X.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=None, random_state=None):
        """
        Split data into training and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config['model_params']['test_size']
        
        if random_state is None:
            random_state = self.config['model_params']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, scaler_path=None, feature_names_path=None):
        """
        Save scaler and feature names.
        
        Args:
            scaler_path: Path to save scaler
            feature_names_path: Path to save feature names
        """
        if scaler_path is None:
            scaler_path = get_absolute_path(self.config['model']['scaler_path'])
        
        if feature_names_path is None:
            feature_names_path = get_absolute_path(
                self.config['model']['feature_names_path']
            )
        
        # Ensure directory exists
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature names
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        logger.info(f"Feature names saved to {feature_names_path}")
    
    def load_artifacts(self, scaler_path=None, feature_names_path=None):
        """
        Load scaler and feature names.
        
        Args:
            scaler_path: Path to load scaler from
            feature_names_path: Path to load feature names from
        """
        if scaler_path is None:
            scaler_path = get_absolute_path(self.config['model']['scaler_path'])
        
        if feature_names_path is None:
            feature_names_path = get_absolute_path(
                self.config['model']['feature_names_path']
            )
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load feature names
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        logger.info(f"Feature names loaded from {feature_names_path}")


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module ready for use.")
    print("Import this module and use FeatureEngineer class for feature preparation.")
