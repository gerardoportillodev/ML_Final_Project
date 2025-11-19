"""Model evaluation module for Credit Risk ML project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from pathlib import Path
from src.logger import setup_logger
from src.config_loader import load_config, get_absolute_path
from src.train_model import ModelTrainer
from src.feature_engineering import FeatureEngineer


logger = setup_logger(__name__)


class ModelEvaluator:
    """Class for evaluating credit risk classification models."""
    
    def __init__(self, config=None):
        """
        Initialize ModelEvaluator.
        
        Args:
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config if config else load_config()
        self.model = None
        self.predictions = None
        self.predictions_proba = None
        
    def load_model(self, model_path=None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Model: Loaded model
        """
        trainer = ModelTrainer(self.config)
        self.model = trainer.load_model(model_path)
        logger.info("Model loaded for evaluation")
        return self.model
    
    def predict(self, X):
        """
        Make predictions on the test data.
        
        Args:
            X: Test features
            
        Returns:
            tuple: (predictions, prediction_probabilities)
        """
        if self.model is None:
            logger.error("No model loaded. Load a model first.")
            raise ValueError("No model loaded")
        
        self.predictions = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            self.predictions_proba = self.model.predict_proba(X)[:, 1]
        else:
            self.predictions_proba = self.predictions
        
        logger.info("Predictions generated")
        return self.predictions, self.predictions_proba
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        logger.info("Evaluation metrics calculated")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
        
        return fpr, tpr, thresholds, roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.close()
        
        return precision, recall, thresholds
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred)
        logger.info(f"Classification Report:\n{report}")
        return report
    
    def evaluate_model(self, X_test, y_test, output_dir=None):
        """
        Complete evaluation of the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save evaluation results
            
        Returns:
            dict: Dictionary with all evaluation results
        """
        if self.model is None:
            logger.error("No model loaded. Load a model first.")
            raise ValueError("No model loaded")
        
        logger.info("=" * 50)
        logger.info("Starting model evaluation")
        logger.info("=" * 50)
        
        # Make predictions
        y_pred, y_pred_proba = self.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        report = self.generate_classification_report(y_test, y_pred)
        
        # Plot confusion matrix
        if output_dir:
            cm_path = Path(output_dir) / "confusion_matrix.png"
            cm = self.plot_confusion_matrix(y_test, y_pred, cm_path)
        else:
            cm = self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot ROC curve
        if y_pred_proba is not None:
            if output_dir:
                roc_path = Path(output_dir) / "roc_curve.png"
                roc_results = self.plot_roc_curve(y_test, y_pred_proba, roc_path)
            else:
                roc_results = self.plot_roc_curve(y_test, y_pred_proba)
            
            # Plot Precision-Recall curve
            if output_dir:
                pr_path = Path(output_dir) / "precision_recall_curve.png"
                pr_results = self.plot_precision_recall_curve(y_test, y_pred_proba, pr_path)
            else:
                pr_results = self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        logger.info("=" * 50)
        logger.info("Model evaluation completed")
        logger.info("=" * 50)
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        return results


def evaluate_pipeline(test_data_path=None, output_dir='evaluation_results'):
    """
    Complete evaluation pipeline.
    
    Args:
        test_data_path: Path to test data file
        output_dir: Directory to save evaluation results
    """
    config = load_config()
    evaluator = ModelEvaluator(config)
    
    # Load model
    evaluator.load_model()
    
    # Load test data
    if test_data_path is None:
        test_data_path = get_absolute_path(config['data']['test_data_path'])
    
    test_data = pd.read_csv(test_data_path)
    logger.info(f"Loaded test data from {test_data_path}")
    
    # Separate features and target
    target_column = config['features'].get('target_column', 'target')
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # Evaluate model
    output_dir = get_absolute_path(output_dir)
    results = evaluator.evaluate_model(X_test, y_test, output_dir)
    
    return results


if __name__ == "__main__":
    try:
        # Run evaluation pipeline
        results = evaluate_pipeline()
        
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        print("\nMetrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        print("=" * 50)
        print("\nEvaluation plots saved in 'evaluation_results' directory")
        
    except FileNotFoundError:
        print("\nError: Test data or model file not found.")
        print("Please run train_model.py first to train the model.")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
