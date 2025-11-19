# Data Directory

This directory contains the data files for the Credit Risk ML project.

## Directory Structure

```
data/
├── raw/              # Raw, unprocessed data files
├── processed/        # Cleaned and processed data files
```

## Expected Data Format

### base_historica.csv

The main input file should be placed in `data/raw/base_historica.csv`.

**Expected structure:**
- The CSV file should contain credit risk data
- It should include a target column (e.g., 'default', 'risk', 'bad_loan', etc.)
- The dataset can contain:
  - Numerical features (age, income, credit_score, loan_amount, etc.)
  - Categorical features (employment_status, education, purpose, etc.)
  - Binary target variable indicating default/no-default

**Example columns:**
- Customer demographics (age, gender, marital_status, etc.)
- Financial information (income, debt, credit_score, etc.)
- Loan details (amount, term, interest_rate, etc.)
- Target variable (default: 0 or 1)

## Configuration

The target column name and other data-related settings can be configured in `config/config.yaml`:

```yaml
features:
  target_column: "default"  # Change this to match your target column name
```

## Data Processing

1. **Raw Data** (`data/raw/`): Place your original CSV file here
2. **Processed Data** (`data/processed/`): Generated automatically by the pipeline
   - `processed_data.csv`: Cleaned data with duplicates removed and missing values handled
   - `train.csv`: Training data split
   - `test.csv`: Test data split

## Privacy and Security

⚠️ **Important**: This directory is included in `.gitignore` to prevent sensitive data from being committed to the repository. Make sure not to commit any data files containing personal or sensitive information.

## Getting Started

If you don't have the `base_historica.csv` file yet:

1. Obtain the credit risk dataset
2. Place it in `data/raw/base_historica.csv`
3. Update the target column name in `config/config.yaml` if needed
4. Run the data processing pipeline: `python src/data_loader.py`
