"""
Model training and evaluation module for loan approval prediction.
Implements various machine learning models and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple
import joblib


class ModelTrainer:
    """
    A class to handle model training, evaluation, and hyperparameter tuning.
    """
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Support Vector Machine': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        
        self.hyperparameter_grids = {
            'Logistic Regression': {
                'C': np.logspace(-4, 4, 20),
                'solver': ['liblinear']
            },
            'Support Vector Machine': {
                'C': [0.25, 0.50, 0.75, 1],
                'kernel': ['linear']
            },
            'Random Forest': {
                'n_estimators': np.arange(10, 1000, 10),
                'max_features': ['log2', 'sqrt'],
                'max_depth': [None, 3, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 20, 50, 100],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        }
        
        self.best_models = {}
        self.model_scores = {}
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> float:
        """
        Evaluate a model using train-test split and cross-validation.
        
        Args:
            model: Scikit-learn model instance
            X: Features
            y: Target variable
            model_name: Name of the model for display
            
        Returns:
            Cross-validation score
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_score = np.mean(cv_scores)
        
        print(f"{model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Cross-Validation Score: {cv_score:.4f}")
        print(f"  CV Standard Deviation: {np.std(cv_scores):.4f}")
        print("-" * 50)
        
        return cv_score
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Dictionary of model names and their cross-validation scores
        """
        print("Comparing Models:")
        print("=" * 50)
        
        model_scores = {}
        
        for name, model in self.models.items():
            cv_score = self.evaluate_model(model, X, y, name)
            model_scores[name] = cv_score
        
        # Sort models by performance
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nModel Rankings:")
        print("=" * 50)
        for rank, (name, score) in enumerate(sorted_models, 1):
            print(f"{rank}. {name}: {score:.4f}")
        
        self.model_scores = model_scores
        return model_scores
    
    def tune_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to tune
            X: Features
            y: Target variable
            
        Returns:
            Best estimator after hyperparameter tuning
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in available models")
        
        if model_name not in self.hyperparameter_grids:
            print(f"No hyperparameter grid defined for {model_name}")
            return self.models[model_name]
        
        print(f"Tuning hyperparameters for {model_name}...")
        print("-" * 50)
        
        model = self.models[model_name]
        param_grid = self.hyperparameter_grids[model_name]
        
        # Perform randomized search
        search = RandomizedSearchCV(
            model, 
            param_grid, 
            n_iter=20, 
            cv=5, 
            verbose=1, 
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        print(f"Best Score for {model_name}: {search.best_score_:.4f}")
        print(f"Best Parameters for {model_name}: {search.best_params_}")
        print("-" * 50)
        
        self.best_models[model_name] = search.best_estimator_
        return search.best_estimator_
    
    def get_detailed_evaluation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Get detailed evaluation metrics for a model.
        
        Args:
            model: Trained model
            X: Features
            y: Target variable
            
        Returns:
            Dictionary containing detailed evaluation metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            model: Trained model to save
            filepath: Path to save the model
        """
        try:
            joblib.dump(model, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> Any:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
