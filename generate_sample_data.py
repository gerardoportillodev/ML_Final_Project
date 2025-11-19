"""
Generate synthetic credit risk data for testing purposes.
This script creates a sample dataset that can be used to test the ML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples=1000, output_path='data/raw/base_historica.csv'):
    """
    Generate synthetic credit risk data for testing.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the generated CSV file
    """
    np.random.seed(42)
    
    print(f"Generating {n_samples} synthetic credit risk samples...")
    
    # Generate features
    data = {
        # Demographics
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'marital_status': np.random.choice(
            ['Single', 'Married', 'Divorced', 'Widowed'], 
            n_samples,
            p=[0.3, 0.5, 0.15, 0.05]
        ),
        'dependents': np.random.randint(0, 5, n_samples),
        'education': np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            n_samples,
            p=[0.3, 0.45, 0.2, 0.05]
        ),
        
        # Financial information
        'income': np.random.randint(20000, 150000, n_samples),
        'employment_status': np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed'],
            n_samples,
            p=[0.7, 0.2, 0.1]
        ),
        'employment_length': np.random.randint(0, 30, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'savings_account': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'checking_account': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
        
        # Loan information
        'loan_amount': np.random.randint(5000, 100000, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60, 84], n_samples),
        'interest_rate': np.random.uniform(3.0, 15.0, n_samples),
        'loan_purpose': np.random.choice(
            ['Education', 'Home', 'Car', 'Business', 'Personal', 'Medical'],
            n_samples,
            p=[0.15, 0.30, 0.20, 0.15, 0.15, 0.05]
        ),
        'property_area': np.random.choice(
            ['Urban', 'Suburban', 'Rural'],
            n_samples,
            p=[0.5, 0.35, 0.15]
        ),
    }
    
    # Calculate debt-to-income ratio
    data['debt_to_income'] = (data['loan_amount'] / data['loan_term']) / (data['income'] / 12)
    data['debt_to_income'] = np.clip(data['debt_to_income'], 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (default) based on features
    # Higher risk for:
    # - Low credit score
    # - High debt-to-income ratio
    # - Unemployed
    # - Low income
    
    risk_score = (
        (850 - df['credit_score']) / 550 * 0.4 +  # Credit score factor
        df['debt_to_income'] * 0.3 +  # Debt ratio factor
        (df['employment_status'] == 'Unemployed').astype(int) * 0.2 +  # Employment factor
        (100000 - df['income']) / 80000 * 0.1  # Income factor
    )
    
    # Add some randomness
    risk_score += np.random.uniform(-0.1, 0.1, n_samples)
    
    # Convert to binary target (default/no default)
    # Use threshold to create imbalanced dataset (more non-defaults)
    threshold = 0.65
    df['default'] = (risk_score > threshold).astype(int)
    
    # Add some missing values randomly (5% of cells)
    for col in df.columns:
        if col != 'default':  # Don't add missing values to target
            mask = np.random.random(n_samples) < 0.05
            if df[col].dtype in ['int64', 'float64']:
                df.loc[mask, col] = np.nan
    
    # Summary statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print(f"Number of defaults: {df['default'].sum()} ({df['default'].mean()*100:.1f}%)")
    print(f"Number of non-defaults: {(df['default']==0).sum()} ({(df['default']==0).mean()*100:.1f}%)")
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Synthetic data saved to: {output_path}")
    print("\nYou can now run the ML pipeline:")
    print("  python main.py --all")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic credit risk data')
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/base_historica.csv',
        help='Output file path (default: data/raw/base_historica.csv)'
    )
    
    args = parser.parse_args()
    
    # Generate data
    generate_sample_data(n_samples=args.samples, output_path=args.output)
