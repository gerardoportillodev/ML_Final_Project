"""
Main runner script for Credit Risk ML Project.
This script provides a command-line interface to run the complete pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger
from src.data_loader import DataLoader
from src.train_model import train_pipeline
from src.evaluate_model import evaluate_pipeline


logger = setup_logger(__name__)


def run_data_processing():
    """Run data loading and cleaning."""
    logger.info("Starting data processing...")
    
    try:
        loader = DataLoader()
        data = loader.load_data()
        info = loader.get_data_info()
        
        print("\nData Info:")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        
        cleaned_data = loader.clean_data()
        loader.save_processed_data()
        
        print("\n✓ Data processing completed successfully!")
        return True
        
    except FileNotFoundError:
        print("\n✗ Error: data file not found.")
        print("Please place 'base_historica.csv' in the data/raw/ directory.")
        return False
    except Exception as e:
        print(f"\n✗ Error during data processing: {str(e)}")
        return False


def run_training(model_type='random_forest'):
    """Run model training."""
    logger.info(f"Starting model training with {model_type}...")
    
    try:
        model, cv_scores = train_pipeline(model_type=model_type)
        
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Model Type: {model_type}")
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("=" * 50)
        print("\n✓ Model training completed successfully!")
        return True
        
    except FileNotFoundError:
        print("\n✗ Error: Processed data file not found.")
        print("Please run data processing first: python main.py --process")
        return False
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run model evaluation."""
    logger.info("Starting model evaluation...")
    
    try:
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
        print("\n✓ Model evaluation completed successfully!")
        return True
        
    except FileNotFoundError:
        print("\n✗ Error: Test data or model file not found.")
        print("Please run training first: python main.py --train")
        return False
    except Exception as e:
        print(f"\n✗ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline(model_type='random_forest'):
    """Run the complete pipeline."""
    print("\n" + "=" * 70)
    print("Running Complete Credit Risk ML Pipeline")
    print("=" * 70)
    
    # Step 1: Data Processing
    print("\n[1/3] Data Processing")
    print("-" * 70)
    if not run_data_processing():
        return False
    
    # Step 2: Model Training
    print("\n[2/3] Model Training")
    print("-" * 70)
    if not run_training(model_type):
        return False
    
    # Step 3: Model Evaluation
    print("\n[3/3] Model Evaluation")
    print("-" * 70)
    if not run_evaluation():
        return False
    
    print("\n" + "=" * 70)
    print("✓ Complete pipeline executed successfully!")
    print("=" * 70)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Credit Risk ML Project - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --all
  
  # Run individual steps
  python main.py --process
  python main.py --train
  python main.py --evaluate
  
  # Train with different model
  python main.py --train --model logistic_regression
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run the complete pipeline (process, train, evaluate)'
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='Run data processing only'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run model training only'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run model evaluation only'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'logistic_regression'],
        help='Model type to train (default: random_forest)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.all, args.process, args.train, args.evaluate]):
        parser.print_help()
        return
    
    # Run requested operations
    success = True
    
    if args.all:
        success = run_full_pipeline(args.model)
    else:
        if args.process:
            success = run_data_processing() and success
        if args.train:
            success = run_training(args.model) and success
        if args.evaluate:
            success = run_evaluation() and success
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
