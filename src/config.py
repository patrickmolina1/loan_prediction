"""
Configuration file for the loan approval prediction project.
Contains constants, file paths, and configuration settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# File paths
DATA_FILE = "loan_data.csv"
MODEL_FILE = "loan_status_predictor.pkl"
SCALER_FILE = "scaler.pkl"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature columns
NUMERICAL_COLUMNS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CATEGORICAL_COLUMNS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

# Encoding mappings
ENCODING_MAP = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4, '4': 4},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
    'Loan_Status': {'Y': 1, 'N': 0}
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Loan Approval Prediction API"
API_DESCRIPTION = "API for predicting loan approval status based on applicant information"
API_VERSION = "1.0.0"

# Sample data for testing
SAMPLE_DATA = {
    'approved': {
        'Gender': 1,  # Male
        'Married': 1,  # Yes
        'Dependents': 0,  # 0
        'Education': 1,  # Graduate
        'Self_Employed': 0,  # No
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,  # Good credit
        'Property_Area': 2  # Urban
    },
    'rejected': {
        'Gender': 0,  # Female
        'Married': 0,  # No
        'Dependents': 4,  # 3+
        'Education': 0,  # Not Graduate
        'Self_Employed': 1,  # Yes
        'ApplicantIncome': 1000,
        'CoapplicantIncome': 0,
        'LoanAmount': 500,
        'Loan_Amount_Term': 180,
        'Credit_History': 0,  # Poor credit
        'Property_Area': 0  # Rural
    }
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project."""
    directories = [DATA_DIR, MODELS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully!")
