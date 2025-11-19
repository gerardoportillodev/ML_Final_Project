"""Credit Risk ML Project - Source Package."""

__version__ = "1.0.0"
__author__ = "ML Team"

from src.config_loader import load_config, get_absolute_path, get_project_root
from src.logger import setup_logger, get_logger
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator

__all__ = [
    'load_config',
    'get_absolute_path',
    'get_project_root',
    'setup_logger',
    'get_logger',
    'DataLoader',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
]
