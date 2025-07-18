"""
Utility functions for the loan approval prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import joblib


def create_sample_data(data_type: str = "approved") -> pd.DataFrame:
    """
    Create sample data for testing.
    
    Args:
        data_type: Type of sample data ('approved' or 'rejected')
        
    Returns:
        DataFrame with sample data
    """
    if data_type == "approved":
        sample = {
            'Gender': [1],  # Male
            'Married': [1],  # Yes
            'Dependents': [0],  # 0
            'Education': [1],  # Graduate
            'Self_Employed': [0],  # No
            'ApplicantIncome': [5000],
            'CoapplicantIncome': [2000],
            'LoanAmount': [150],
            'Loan_Amount_Term': [360],
            'Credit_History': [1],  # Good credit
            'Property_Area': [2]  # Urban
        }
    else:  # rejected
        sample = {
            'Gender': [0],  # Female
            'Married': [0],  # No
            'Dependents': [4],  # 3+
            'Education': [0],  # Not Graduate
            'Self_Employed': [1],  # Yes
            'ApplicantIncome': [1000],
            'CoapplicantIncome': [0],
            'LoanAmount': [500],
            'Loan_Amount_Term': [180],
            'Credit_History': [0],  # Poor credit
            'Property_Area': [0]  # Rural
        }
    
    return pd.DataFrame(sample)


def decode_prediction(prediction: int) -> str:
    """
    Decode numerical prediction to human-readable format.
    
    Args:
        prediction: Numerical prediction (0 or 1)
        
    Returns:
        String representation of the prediction
    """
    return "Approved" if prediction == 1 else "Rejected"


def print_data_info(df: pd.DataFrame) -> None:
    """
    Print comprehensive information about the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("Dataset Information:")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn Information:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nNumerical Columns Summary:")
    print(df.describe())


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 10) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def save_preprocessing_objects(scaler: Any, filepath: str) -> None:
    """
    Save preprocessing objects (scaler) to file.
    
    Args:
        scaler: Fitted scaler object
        filepath: Path to save the scaler
    """
    try:
        joblib.dump(scaler, filepath)
        print(f"Scaler saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving scaler: {e}")
        raise


def load_preprocessing_objects(filepath: str) -> Any:
    """
    Load preprocessing objects from file.
    
    Args:
        filepath: Path to the saved scaler
        
    Returns:
        Loaded scaler object
    """
    try:
        scaler = joblib.load(filepath)
        print(f"Scaler loaded successfully from {filepath}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        raise


def validate_input_data(data: Dict[str, Any]) -> bool:
    """
    Validate input data for prediction.
    
    Args:
        data: Dictionary containing input data
        
    Returns:
        Boolean indicating if data is valid
    """
    required_fields = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            print(f"Missing required field: {field}")
            return False
    
    # Check data types and ranges
    try:
        # Categorical fields should be 0 or 1 (mostly)
        if data['Gender'] not in [0, 1]:
            print("Gender should be 0 (Female) or 1 (Male)")
            return False
        
        if data['Married'] not in [0, 1]:
            print("Married should be 0 (No) or 1 (Yes)")
            return False
        
        if data['Dependents'] not in [0, 1, 2, 4]:
            print("Dependents should be 0, 1, 2, or 4")
            return False
        
        # Check if numerical fields are positive
        numerical_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for field in numerical_fields:
            if data[field] < 0:
                print(f"{field} should be non-negative")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False
