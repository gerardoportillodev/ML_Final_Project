"""Unsupervised Learning Project - Source Package."""

__version__ = "2.0.0"
__author__ = "ML Team"

from src.config_loader import load_config, get_absolute_path, get_project_root
from src.logger import setup_logger, get_logger
from src.data_loading import DataLoader
from src.preprocessing import DataPreprocessor
from src.dim_reduction import DimensionalityReducer
from src.clustering import ClusteringAnalyzer
from src.evaluation import ClusteringEvaluator

__all__ = [
    'load_config',
    'get_absolute_path',
    'get_project_root',
    'setup_logger',
    'get_logger',
    'DataLoader',
    'DataPreprocessor',
    'DimensionalityReducer',
    'ClusteringAnalyzer',
    'ClusteringEvaluator',
]
