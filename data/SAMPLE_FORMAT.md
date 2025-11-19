# Sample Data Format for Credit Risk Project

This document describes the expected format for the `base_historica.csv` file.

## Expected Columns

The credit risk dataset should contain the following types of columns:

### Target Variable (Required)
- **default** or **risk** or **bad_loan**: Binary variable (0 or 1) indicating whether the customer defaulted
  - 0 = No default (good customer)
  - 1 = Default (bad customer)

### Customer Demographics (Optional but Recommended)
- **age**: Customer age (numerical)
- **gender**: Customer gender (categorical: Male/Female)
- **marital_status**: Marital status (categorical: Single/Married/Divorced/Widowed)
- **education**: Education level (categorical: High School/Bachelor/Master/PhD)
- **dependents**: Number of dependents (numerical)

### Financial Information (Recommended)
- **income**: Annual or monthly income (numerical)
- **employment_status**: Employment status (categorical: Employed/Unemployed/Self-Employed)
- **employment_length**: Years with current employer (numerical)
- **debt_to_income**: Debt-to-income ratio (numerical)
- **credit_score**: Credit score (numerical, e.g., 300-850)
- **savings_account**: Has savings account (categorical: Yes/No or binary 0/1)
- **checking_account**: Has checking account (categorical: Yes/No or binary 0/1)

### Loan Information (Recommended)
- **loan_amount**: Requested loan amount (numerical)
- **loan_term**: Loan term in months (numerical)
- **interest_rate**: Interest rate (numerical)
- **loan_purpose**: Purpose of loan (categorical: Education/Home/Car/Business/Personal/Medical)
- **property_area**: Property location (categorical: Urban/Suburban/Rural)

## Example Data Structure

```csv
age,income,credit_score,employment_status,loan_amount,loan_term,debt_to_income,education,default
35,50000,720,Employed,20000,36,0.35,Bachelor,0
42,75000,680,Employed,35000,60,0.42,Master,0
28,30000,650,Self-Employed,15000,24,0.50,Bachelor,1
55,90000,750,Employed,50000,84,0.28,PhD,0
31,40000,600,Unemployed,10000,12,0.65,High School,1
```

## Important Notes

1. **Column Names**: The exact column names can vary. Update `config/config.yaml` to match your data:
   ```yaml
   features:
     target_column: "default"  # Change to match your target column name
   ```

2. **Missing Values**: The pipeline handles missing values, but minimize them for best results.

3. **Categorical Encoding**: Categorical features are automatically encoded by the pipeline.

4. **Data Types**: Ensure numerical columns contain numeric values and categorical columns contain text.

5. **File Format**: Must be a valid CSV file with comma separators.

## Minimum Requirements

At minimum, your dataset should have:
- At least one target column (binary: 0/1)
- At least 3-5 feature columns (mix of numerical and/or categorical)
- At least 100 rows (more is better, ideally 1000+)

## Getting Sample Data

If you don't have real credit risk data, you can:
1. Use publicly available credit datasets (e.g., from Kaggle, UCI ML Repository)
2. Generate synthetic data for testing
3. Contact your instructor for the course dataset

## Privacy Reminder

⚠️ **Never commit real customer data to version control!** The `.gitignore` file is configured to exclude CSV files from the repository.
