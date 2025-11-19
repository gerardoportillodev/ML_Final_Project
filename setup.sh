#!/bin/bash
# Setup script for Credit Risk ML Project

echo "=========================================="
echo "Credit Risk ML Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Creating necessary directories..."
mkdir -p data/raw data/processed models logs evaluation_results

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Place your data file in: data/raw/base_historica.csv"
echo "3. Run data processing: python src/data_loader.py"
echo "4. Train the model: python src/train_model.py"
echo "5. Evaluate the model: python src/evaluate_model.py"
echo ""
echo "For more information, see README.md"
