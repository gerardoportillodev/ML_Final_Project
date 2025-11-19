# Quick Start Guide

This guide will help you get started with the Credit Risk ML Project in just a few steps.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation (5 minutes)

### Option 1: Automatic Setup (Recommended)

**On Linux/Mac:**
```bash
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick Usage

### Option 1: Run Complete Pipeline (Recommended for First Time)

```bash
# Activate virtual environment first!
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the complete pipeline
python main.py --all
```

**Note:** You need to place your `base_historica.csv` file in `data/raw/` directory first.

### Option 2: Run Step by Step

```bash
# 1. Process the data
python main.py --process

# 2. Train the model
python main.py --train

# 3. Evaluate the model
python main.py --evaluate
```

### Option 3: Use Individual Scripts

```bash
# Process data
python src/data_loader.py

# Train model
python src/train_model.py

# Evaluate model
python src/evaluate_model.py
```

## Expected Output

After running the complete pipeline, you should see:

1. **Processed Data**: `data/processed/processed_data.csv`
2. **Trained Model**: `models/credit_risk_model.pkl`
3. **Evaluation Results**: 
   - `evaluation_results/confusion_matrix.png`
   - `evaluation_results/roc_curve.png`
   - `evaluation_results/precision_recall_curve.png`
4. **Logs**: `logs/credit_risk_ml.log`

## Exploratory Data Analysis

To run the interactive EDA notebook:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch Jupyter
jupyter notebook notebooks/eda_credit_risk.ipynb
```

## Common Issues and Solutions

### Issue 1: Data File Not Found

**Error:** `FileNotFoundError: data/raw/base_historica.csv not found`

**Solution:** Place your CSV file in the `data/raw/` directory:
```bash
# Make sure the file exists
ls data/raw/base_historica.csv
```

### Issue 2: Module Import Error

**Error:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:** Make sure you've activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 3: Wrong Target Column Name

**Error:** `Target column 'default' not found in dataset`

**Solution:** Update the target column name in `config/config.yaml`:
```yaml
features:
  target_column: "your_actual_column_name"
```

## Running Tests

To verify everything is working correctly:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## Project Configuration

Edit `config/config.yaml` to customize:

- Data file paths
- Model parameters (test size, random state, CV folds)
- Logging settings
- Target column name

Example:
```yaml
model_params:
  test_size: 0.3        # Change test set size
  random_state: 123     # Change random seed
  cv_folds: 10          # Change number of CV folds
```

## Next Steps

1. **Explore the Data**: Use the Jupyter notebook to understand your data better
2. **Experiment with Models**: Try both Random Forest and Logistic Regression
3. **Tune Parameters**: Modify model parameters in the training scripts
4. **Add Features**: Extend the feature engineering module with domain-specific features
5. **Compare Models**: Train multiple models and compare their performance

## Getting Help

For more detailed information, see:
- `README.md` - Complete project documentation
- `data/SAMPLE_FORMAT.md` - Expected data format
- `src/` modules - Each module has detailed docstrings

## Example Workflow

```bash
# 1. Setup (one time)
./setup.sh

# 2. Place your data
cp path/to/your/data.csv data/raw/base_historica.csv

# 3. Run complete pipeline
source venv/bin/activate
python main.py --all

# 4. Review results
ls evaluation_results/
cat logs/credit_risk_ml.log

# 5. Explore in notebook
jupyter notebook notebooks/eda_credit_risk.ipynb
```

## Performance Benchmarks

On a typical laptop (4 cores, 8GB RAM), expect:
- Data processing: < 1 minute for 100k rows
- Model training: 1-5 minutes depending on model
- Model evaluation: < 30 seconds

## Tips for Best Results

1. **Data Quality**: Clean your data thoroughly before training
2. **Feature Engineering**: Add domain-specific features that make sense for credit risk
3. **Model Selection**: Try both Random Forest and Logistic Regression, compare results
4. **Cross-Validation**: Use CV scores to assess model stability
5. **Threshold Tuning**: Adjust classification threshold based on business needs

---

**Happy Modeling! 🚀**

For issues or questions, refer to the full README.md documentation.
