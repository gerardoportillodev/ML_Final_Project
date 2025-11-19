# Credit Risk ML Project

A comprehensive Machine Learning project for credit risk prediction using classification algorithms. This project implements a complete ML pipeline including data loading, preprocessing, feature engineering, model training, and evaluation.

## Project Structure

```
ML_Final_Project/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config_loader.py          # Configuration management
│   ├── logger.py                 # Logging utilities
│   ├── data_loader.py            # Data loading and cleaning
│   ├── feature_engineering.py    # Feature preprocessing and engineering
│   ├── train_model.py            # Model training
│   └── evaluate_model.py         # Model evaluation
├── notebooks/                     # Jupyter notebooks
│   └── eda_credit_risk.ipynb     # Exploratory Data Analysis
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_data_pipeline.py     # Data pipeline tests
├── config/                        # Configuration files
│   └── config.yaml               # Project configuration
├── data/                          # Data directory
│   ├── raw/                      # Raw data files (place base_historica.csv here)
│   └── processed/                # Processed data files
├── models/                        # Saved models
├── logs/                          # Log files
├── evaluation_results/            # Evaluation plots and results
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

- **Data Loading & Cleaning**: Robust data loading with missing value handling and duplicate removal
- **Feature Engineering**: Automated feature identification, encoding, and scaling
- **Model Training**: Support for multiple classification algorithms (Random Forest, Logistic Regression)
- **Model Evaluation**: Comprehensive evaluation with ROC AUC, confusion matrix, and classification reports
- **Logging**: Detailed logging throughout the pipeline
- **Configuration Management**: YAML-based configuration for easy customization
- **Unit Tests**: Test coverage for data pipeline components
- **EDA Notebook**: Interactive exploratory data analysis

## Requirements

- Python 3.8+
- See `requirements.txt` for complete list of dependencies

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gerardoportillodev/ML_Final_Project.git
cd ML_Final_Project
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place your `base_historica.csv` file in the `data/raw/` directory:

```bash
# Make sure the data file is in the correct location
# data/raw/base_historica.csv
```

## Usage

### Step 1: Data Loading and Cleaning

```bash
python src/data_loader.py
```

This script will:
- Load the raw data from `data/raw/base_historica.csv`
- Display data information and statistics
- Clean the data (remove duplicates and handle missing values)
- Save processed data to `data/processed/processed_data.csv`

### Step 2: Model Training

```bash
# Train Random Forest (default)
python src/train_model.py

# Or train Logistic Regression
python src/train_model.py logistic_regression
```

This script will:
- Load the processed data
- Perform feature engineering (encoding, scaling)
- Split data into training and test sets
- Train the classification model
- Perform cross-validation
- Save the trained model and artifacts

### Step 3: Model Evaluation

```bash
python src/evaluate_model.py
```

This script will:
- Load the trained model
- Make predictions on test data
- Calculate evaluation metrics (accuracy, precision, recall, F1, ROC AUC)
- Generate and save evaluation plots:
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
- Display classification report

### Step 4: Exploratory Data Analysis (Optional)

Launch Jupyter notebook for interactive EDA:

```bash
jupyter notebook notebooks/eda_credit_risk.ipynb
```

## Running Tests

Execute the unit tests to verify the data pipeline:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Configuration

Edit `config/config.yaml` to customize:

- Data paths
- Model parameters
- Logging settings
- Feature engineering options

Example configuration:

```yaml
data:
  raw_data_path: "data/raw/base_historica.csv"
  processed_data_path: "data/processed/processed_data.csv"

model_params:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

features:
  target_column: "default"
```

## Model Performance

The project evaluates models using multiple metrics:

- **ROC AUC**: Area under the ROC curve
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

Results are saved in the `evaluation_results/` directory with visualization plots.

## Using the Project as a Library

You can also import and use the modules in your own Python scripts:

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator

# Load and clean data
loader = DataLoader()
data = loader.load_data()
cleaned_data = loader.clean_data()

# Prepare features
engineer = FeatureEngineer()
X, y = engineer.prepare_features(cleaned_data)

# Train model
trainer = ModelTrainer()
model = trainer.train_random_forest(X, y)

# Evaluate model
evaluator = ModelEvaluator()
evaluator.model = model
results = evaluator.evaluate_model(X_test, y_test)
```

## Logging

Logs are automatically saved to the `logs/` directory with timestamps. Log level can be configured in `config/config.yaml`.

## Troubleshooting

### Data File Not Found

Ensure `base_historica.csv` is in the `data/raw/` directory:

```bash
ls data/raw/base_historica.csv
```

### Module Import Errors

Make sure you've activated the virtual environment and installed all dependencies:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Permission Errors

Ensure you have write permissions for the `data/`, `models/`, and `logs/` directories.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is part of an academic assignment for the Machine Learning course.

## Authors

- ML Team - Credit Risk Analysis

## Acknowledgments

- Scikit-learn documentation and examples
- Pandas and NumPy communities
- Machine Learning course materials
