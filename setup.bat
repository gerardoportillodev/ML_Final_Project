@echo off
REM Setup script for Credit Risk ML Project (Windows)

echo ==========================================
echo Credit Risk ML Project Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo.
echo Creating necessary directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "evaluation_results" mkdir evaluation_results

echo.
echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Place your data file in: data\raw\base_historica.csv
echo 3. Run data processing: python src\data_loader.py
echo 4. Train the model: python src\train_model.py
echo 5. Evaluate the model: python src\evaluate_model.py
echo.
echo For more information, see README.md

pause
