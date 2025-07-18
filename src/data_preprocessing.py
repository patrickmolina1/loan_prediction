"""
Data preprocessing module for loan approval prediction.
Handles data cleaning, feature engineering, and preprocessing steps.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any


class DataPreprocessor:
    """
    A class to handle data preprocessing for loan approval prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoding_map = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4, '4': 4},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
            'Loan_Status': {'Y': 1, 'N': 0}
        }
        self.numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Create a copy to avoid modifying original data
        df_cleaned = df.copy()
        
        # Drop unnecessary columns
        if 'Loan_ID' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop("Loan_ID", axis=1)
        
        # Drop rows with critical missing values
        df_cleaned = df_cleaned.dropna(subset=['Gender', 'Dependents', 'Loan_Amount_Term'])
        
        # Fill missing values with mode for categorical variables
        categorical_columns = ['Self_Employed', 'Credit_History']
        for col in categorical_columns:
            if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
                mode_value = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(mode_value, inplace=True)
        
        # Handle special case for Dependents
        if 'Dependents' in df_cleaned.columns:
            df_cleaned['Dependents'].replace('3+', '4', inplace=True)
        
        print(f"Missing values handled. New shape: {df_cleaned.shape}")
        return df_cleaned
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using predefined mapping.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        df_encoded.replace(self.encoding_map, inplace=True)
        print("Categorical variables encoded successfully")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        if fit_scaler:
            df_scaled[self.numerical_columns] = self.scaler.fit_transform(df_scaled[self.numerical_columns])
        else:
            df_scaled[self.numerical_columns] = self.scaler.transform(df_scaled[self.numerical_columns])
        
        print("Numerical features scaled successfully")
        return df_scaled
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        if 'Loan_Status' not in df.columns:
            raise ValueError("Target variable 'Loan_Status' not found in DataFrame")
        
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        
        print(f"Features prepared. Shape: {X.shape}")
        print(f"Target prepared. Shape: {y.shape}")
        
        return X, y
    
    def full_preprocessing_pipeline(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (processed_features, target)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Separate features and target
        X, y = self.prepare_features_target(df)
        
        # Scale numerical features
        X = self.scale_numerical_features(X, fit_scaler=True)
        
        return X, y
    
    def preprocess_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for prediction (without target variable).
        
        Args:
            data: Input DataFrame for prediction
            
        Returns:
            Processed DataFrame ready for prediction
        """
        # Encode categorical variables
        data_encoded = self.encode_categorical_variables(data)
        
        # For prediction, we need to use the same scaler that was used during training
        # If the scaler is not fitted, we'll just return the encoded data
        try:
            # Scale numerical features (without fitting)
            data_scaled = self.scale_numerical_features(data_encoded, fit_scaler=False)
            return data_scaled
        except Exception as e:
            print(f"Warning: Could not scale features - {e}")
            print("Returning unscaled data. Make sure to use the same scaler from training.")
            return data_encoded
