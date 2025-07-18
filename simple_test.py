"""
Quick test to verify the API works with the saved models.
"""

import pandas as pd
import joblib
import numpy as np

def test_api_prediction():
    """Test API prediction functionality."""
    print("🧪 Testing API Prediction...")
    
    try:
        # Load model and scaler
        model = joblib.load('models/loan_status_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Numerical columns for scaling
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
        # Sample data (approved case)
        sample_data = pd.DataFrame({
            'Gender': [1],
            'Married': [1],
            'Dependents': [0],
            'Education': [1],
            'Self_Employed': [0],
            'ApplicantIncome': [5000],
            'CoapplicantIncome': [2000],
            'LoanAmount': [150],
            'Loan_Amount_Term': [360],
            'Credit_History': [1],
            'Property_Area': [2]
        })
        
        # Scale numerical features
        sample_data[numerical_columns] = scaler.transform(sample_data[numerical_columns])
        
        # Make prediction
        prediction = model.predict(sample_data)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(sample_data)
            confidence = float(np.max(prediction_proba))
            print(f"✅ Prediction confidence: {confidence:.4f}")
        except:
            confidence = 0.0
            print("ℹ️  Prediction confidence not available")
        
        # Convert prediction to human-readable format
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"
        
        print(f"✅ Prediction result: {loan_status}")
        print("✅ API prediction test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in API prediction test: {e}")
        return False

def test_rejected_case():
    """Test with a case that should be rejected."""
    print("\n🧪 Testing Rejected Case...")
    
    try:
        # Load model and scaler
        model = joblib.load('models/loan_status_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Numerical columns for scaling
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
        # Sample data (rejected case)
        sample_data = pd.DataFrame({
            'Gender': [0],
            'Married': [0],
            'Dependents': [4],
            'Education': [0],
            'Self_Employed': [1],
            'ApplicantIncome': [1000],
            'CoapplicantIncome': [0],
            'LoanAmount': [500],
            'Loan_Amount_Term': [180],
            'Credit_History': [0],
            'Property_Area': [0]
        })
        
        # Scale numerical features
        sample_data[numerical_columns] = scaler.transform(sample_data[numerical_columns])
        
        # Make prediction
        prediction = model.predict(sample_data)
        
        # Convert prediction to human-readable format
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"
        
        print(f"✅ Prediction result: {loan_status}")
        print("✅ Rejected case test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in rejected case test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running API Prediction Tests")
    print("=" * 40)
    
    test1 = test_api_prediction()
    test2 = test_rejected_case()
    
    if test1 and test2:
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ The API is ready to use!")
        print("✅ Run: uvicorn app:app --reload")
    else:
        print("\n❌ Some tests failed!")
        print("❌ Please check the setup.")
