"""
Preprocessing module for unsupervised learning.
Handles data cleaning, normalization, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for unsupervised learning data preparation."""
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'robust', or 'none'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
    
    def handle_missing_values(self, df, strategy='median', threshold=0.5):
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', or 'most_frequent'
            threshold: Drop columns with more than this fraction of missing values
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric missing values
        if numeric_cols:
            self.imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            logger.info(f"Imputed {len(numeric_cols)} numeric columns")
        
        # Handle categorical missing values
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            logger.info(f"Imputed {len(categorical_cols)} categorical columns")
        
        return df
    
    def remove_outliers(self, df, method='iqr', threshold=3.0):
        """
        Remove outliers from numeric features.
        
        Args:
            df: Input DataFrame
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        initial_rows = len(df)
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        rows_removed = initial_rows - len(df)
        logger.info(f"Removed {rows_removed} rows ({rows_removed/initial_rows*100:.2f}%)")
        
        return df
    
    def encode_categorical(self, df, method='onehot'):
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            method: 'onehot' or 'label'
            
        Returns:
            DataFrame with encoded categorical variables
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns using {method}")
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled DataFrame
        """
        if self.scaler is None or self.scaling_method == 'none':
            logger.info("No scaling applied")
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"Fitted and scaled {len(numeric_cols)} features using {self.scaling_method}")
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info(f"Scaled {len(numeric_cols)} features using fitted scaler")
        
        return df
    
    def preprocess(self, df, remove_outliers_flag=True, encode_categorical_flag=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            remove_outliers_flag: Whether to remove outliers
            encode_categorical_flag: Whether to encode categorical variables
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Initial shape: {df.shape}")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        if remove_outliers_flag:
            df = self.remove_outliers(df)
        
        # Encode categorical variables
        if encode_categorical_flag:
            df = self.encode_categorical(df)
        
        # Scale features
        df = self.scale_features(df, fit=True)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        logger.info(f"Final shape: {df.shape}")
        logger.info("Preprocessing complete")
        
        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Preprocessing module ready for use.")
