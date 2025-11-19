"""Configuration loader for the Credit Risk ML project."""

import os
import yaml
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_absolute_path(relative_path):
    """
    Convert relative path to absolute path from project root.
    
    Args:
        relative_path: Relative path from project root
        
    Returns:
        Path: Absolute path
    """
    project_root = get_project_root()
    return project_root / relative_path
