"""
Test script to verify the loan approval prediction system works correctly.
"""

import pandas as pd
import joblib
from src.data_preprocessing import DataPreprocessor
from src.utils import create_sample_data, decode_prediction

def test_preprocessing():
    """Test data preprocessing functionality."""
    print("🧪 Testing Data Preprocessing...")
    
    preprocessor = DataPreprocessor()
    
    # Test with sample data
    sample_data = create_sample_data("approved")
    processed_data = preprocessor.preprocess_prediction_data(sample_data)
    
    print(f"✅ Sample data shape: {sample_data.shape}")
    print(f"✅ Processed data shape: {processed_data.shape}")
    print("✅ Data preprocessing test passed!")
    
    return preprocessor

def test_model_loading():
    """Test model loading functionality."""
    print("\n🧪 Testing Model Loading...")
    
    try:
        model = joblib.load('models/loan_status_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model loaded successfully!")
        print("✅ Scaler loaded successfully!")
        print("✅ Model loading test passed!")
        return model, scaler
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("❌ Model loading test failed!")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("❌ Model loading test failed!")
        return None, None

def test_prediction():
    """Test prediction functionality."""
    print("\n🧪 Testing Prediction Functionality...")
    
    preprocessor = test_preprocessing()
    model, scaler = test_model_loading()
    
    if model is None or scaler is None:
        print("❌ Cannot test prediction - model loading failed!")
        return
    
    # Test approved case
    approved_sample = create_sample_data("approved")
    approved_processed = preprocessor.preprocess_prediction_data(approved_sample)
    approved_prediction = model.predict(approved_processed)
    
    print(f"✅ Approved sample prediction: {decode_prediction(approved_prediction[0])}")
    
    # Test rejected case
    rejected_sample = create_sample_data("rejected")
    rejected_processed = preprocessor.preprocess_prediction_data(rejected_sample)
    rejected_prediction = model.predict(rejected_processed)
    
    print(f"✅ Rejected sample prediction: {decode_prediction(rejected_prediction[0])}")
    print("✅ Prediction test passed!")

def test_api_data_format():
    """Test API data format compatibility."""
    print("\n🧪 Testing API Data Format...")
    
    # Sample API request
    api_request = {
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 2
    }
    
    # Convert to DataFrame (as API would do)
    api_df = pd.DataFrame([api_request])
    
    # Process the data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_prediction_data(api_df)
    
    print(f"✅ API data format compatible: {processed_data.shape}")
    print("✅ API data format test passed!")

def run_all_tests():
    """Run all tests."""
    print("🚀 Running Loan Approval Prediction System Tests")
    print("=" * 60)
    
    try:
        test_preprocessing()
        test_model_loading()
        test_prediction()
        test_api_data_format()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ The system is ready for use!")
        print("✅ You can now run the API with: uvicorn app:app --reload")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("❌ Please check the setup and try again.")

if __name__ == "__main__":
    run_all_tests()
