"""Model training module for Credit Risk ML project."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path
from src.logger import setup_logger
from src.config_loader import load_config, get_absolute_path
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer


logger = setup_logger(__name__)


class ModelTrainer:
    """Class for training credit risk classification models."""
    
    def __init__(self, config=None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config if config else load_config()
        self.model = None
        self.best_model = None
        self.cv_scores = None
        
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train a Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest classifier")
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.config['model_params']['random_state'],
            'n_jobs': -1
        }
        params.update(kwargs)
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        logger.info("Random Forest training completed")
        logger.info(f"Model parameters: {self.model.get_params()}")
        
        return self.model
    
    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """
        Train a Logistic Regression classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            LogisticRegression: Trained model
        """
        logger.info("Training Logistic Regression classifier")
        
        # Default parameters
        params = {
            'max_iter': 1000,
            'random_state': self.config['model_params']['random_state'],
            'n_jobs': -1
        }
        params.update(kwargs)
        
        self.model = LogisticRegression(**params)
        self.model.fit(X_train, y_train)
        
        logger.info("Logistic Regression training completed")
        logger.info(f"Model parameters: {self.model.get_params()}")
        
        return self.model
    
    def cross_validate(self, X, y, cv=None, scoring='roc_auc'):
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            np.ndarray: Cross-validation scores
        """
        if self.model is None:
            logger.error("No model trained. Train a model first.")
            raise ValueError("No model trained")
        
        if cv is None:
            cv = self.config['model_params']['cv_folds']
        
        logger.info(f"Performing {cv}-fold cross-validation")
        
        self.cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        logger.info(f"Cross-validation {scoring} scores: {self.cv_scores}")
        logger.info(f"Mean {scoring}: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        
        return self.cv_scores
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        if self.model is None:
            logger.error("No model trained. Train a model first.")
            raise ValueError("No model trained")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance calculated")
        logger.info(f"Top 5 features:\n{importance_df.head()}")
        
        return importance_df
    
    def save_model(self, model_path=None):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model. If None, uses path from config.
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            raise ValueError("No model to save")
        
        if model_path is None:
            model_path = get_absolute_path(self.config['model']['model_path'])
        
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path=None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to load the model from. If None, uses path from config.
            
        Returns:
            Model: Loaded model
        """
        if model_path is None:
            model_path = get_absolute_path(self.config['model']['model_path'])
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return self.model


def train_pipeline(data_path=None, model_type='random_forest'):
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to processed data file
        model_type: Type of model to train ('random_forest' or 'logistic_regression')
    """
    config = load_config()
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    
    # Load data
    logger.info("=" * 50)
    logger.info("Starting training pipeline")
    logger.info("=" * 50)
    
    if data_path is None:
        data_path = get_absolute_path(config['data']['processed_data_path'])
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data from {data_path}")
    
    # Prepare features
    X, y = feature_engineer.prepare_features(data, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = feature_engineer.split_data(X, y)
    
    # Train model
    if model_type == 'random_forest':
        model = trainer.train_random_forest(X_train, y_train)
    elif model_type == 'logistic_regression':
        model = trainer.train_logistic_regression(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation
    cv_scores = trainer.cross_validate(X_train, y_train)
    
    # Get feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importance = trainer.get_feature_importance(feature_engineer.feature_names)
    
    # Save model and artifacts
    trainer.save_model()
    feature_engineer.save_artifacts()
    
    # Save train and test data
    train_data_path = get_absolute_path(config['data']['train_data_path'])
    test_data_path = get_absolute_path(config['data']['test_data_path'])
    
    X_train['target'] = y_train
    X_test['target'] = y_test
    
    X_train.to_csv(train_data_path, index=False)
    X_test.to_csv(test_data_path, index=False)
    
    logger.info(f"Train data saved to {train_data_path}")
    logger.info(f"Test data saved to {test_data_path}")
    
    logger.info("=" * 50)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 50)
    
    return model, cv_scores


if __name__ == "__main__":
    import sys
    
    try:
        # Get model type from command line if provided
        model_type = sys.argv[1] if len(sys.argv) > 1 else 'random_forest'
        
        # Run training pipeline
        model, cv_scores = train_pipeline(model_type=model_type)
        
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Model Type: {model_type}")
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("=" * 50)
        
    except FileNotFoundError:
        print("\nError: Processed data file not found.")
        print("Please run data_loader.py first to process the data.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
