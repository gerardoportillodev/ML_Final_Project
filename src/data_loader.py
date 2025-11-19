"""Data loading and cleaning module for Credit Risk ML project."""

import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import setup_logger
from src.config_loader import load_config, get_absolute_path


logger = setup_logger(__name__)


class DataLoader:
    """Class for loading and cleaning credit risk data."""
    
    def __init__(self, config=None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config if config else load_config()
        self.data = None
        
    def load_data(self, file_path=None):
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, uses path from config.
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if file_path is None:
            file_path = get_absolute_path(self.config['data']['raw_data_path'])
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self):
        """
        Get basic information about the loaded data.
        
        Returns:
            dict: Dictionary with data statistics
        """
        if self.data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
        }
        
        logger.info("Data information retrieved")
        return info
    
    def clean_data(self, drop_duplicates=True, handle_missing='drop'):
        """
        Clean the loaded data.
        
        Args:
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: How to handle missing values ('drop', 'fill_mean', 'fill_median')
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            raise ValueError("No data loaded")
        
        logger.info("Starting data cleaning")
        initial_shape = self.data.shape
        
        # Drop duplicates
        if drop_duplicates:
            self.data = self.data.drop_duplicates()
            logger.info(f"Duplicates removed. Rows removed: {initial_shape[0] - self.data.shape[0]}")
        
        # Handle missing values
        if handle_missing == 'drop':
            self.data = self.data.dropna()
            logger.info(f"Missing values dropped. Current shape: {self.data.shape}")
        elif handle_missing == 'fill_mean':
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(
                self.data[numeric_columns].mean()
            )
            logger.info("Missing values filled with mean")
        elif handle_missing == 'fill_median':
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(
                self.data[numeric_columns].median()
            )
            logger.info("Missing values filled with median")
        
        logger.info(f"Data cleaning completed. Final shape: {self.data.shape}")
        return self.data
    
    def save_processed_data(self, output_path=None):
        """
        Save processed data to CSV file.
        
        Args:
            output_path: Path to save the file. If None, uses path from config.
        """
        if self.data is None:
            logger.error("No data to save. Load and clean data first.")
            raise ValueError("No data to save")
        
        if output_path is None:
            output_path = get_absolute_path(self.config['data']['processed_data_path'])
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    try:
        # Load data
        data = loader.load_data()
        
        # Get data info
        info = loader.get_data_info()
        print("\nData Info:")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        
        # Clean data
        cleaned_data = loader.clean_data()
        
        # Save processed data
        loader.save_processed_data()
        
        print("\nData processing completed successfully!")
        
    except FileNotFoundError:
        print("\nError: data file not found. Please place 'base_historica.csv' in the data/raw/ directory.")
    except Exception as e:
        print(f"\nError during data processing: {str(e)}")
