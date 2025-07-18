"""
FastAPI application for loan approval prediction.
Provides REST API endpoints for real-time loan status prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status based on applicant information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load('models/loan_status_predictor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model or scaler file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Define numerical columns for scaling
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

class LoanApplication(BaseModel):
    """
    Pydantic model for loan application data validation.
    """
    Gender: int = Field(..., ge=0, le=1, description="Gender (0: Female, 1: Male)")
    Married: int = Field(..., ge=0, le=1, description="Marital status (0: No, 1: Yes)")
    Dependents: int = Field(..., ge=0, le=4, description="Number of dependents (0, 1, 2, 4)")
    Education: int = Field(..., ge=0, le=1, description="Education (0: Not Graduate, 1: Graduate)")
    Self_Employed: int = Field(..., ge=0, le=1, description="Self employment status (0: No, 1: Yes)")
    ApplicantIncome: float = Field(..., ge=0, description="Applicant income")
    CoapplicantIncome: float = Field(..., ge=0, description="Co-applicant income")
    LoanAmount: float = Field(..., ge=0, description="Loan amount in thousands")
    Loan_Amount_Term: float = Field(..., ge=0, description="Loan term in months")
    Credit_History: int = Field(..., ge=0, le=1, description="Credit history (0: Bad, 1: Good)")
    Property_Area: int = Field(..., ge=0, le=2, description="Property area (0: Rural, 1: Semiurban, 2: Urban)")

    @validator('Dependents')
    def validate_dependents(cls, v):
        if v not in [0, 1, 2, 4]:
            raise ValueError('Dependents must be 0, 1, 2, or 4')
        return v

    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    loan_status: str = Field(..., description="Predicted loan status")
    confidence: float = Field(..., description="Prediction confidence score")
    application_id: str = Field(..., description="Unique application ID")

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint."""
    try:
        # Test if model and scaler are accessible
        _ = model.get_params()
        _ = scaler.get_params()
        return {
            "status": "healthy",
            "model_loaded": True,
            "scaler_loaded": True,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/predict", response_model=Dict[str, str], tags=["Prediction"])
async def predict_loan_status(application: LoanApplication):
    """
    Predict loan approval status based on applicant information.
    
    Args:
        application: LoanApplication object containing applicant data
        
    Returns:
        Dictionary containing prediction result
    """
    try:
        # Convert application to DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Scale numerical features
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get prediction probability for confidence
        try:
            prediction_proba = model.predict_proba(input_data)
            confidence = float(np.max(prediction_proba))
        except AttributeError:
            # Some models don't have predict_proba
            confidence = 0.0
        
        # Convert prediction to human-readable format
        loan_status = "Approved" if prediction[0] == 1 else "Rejected"
        
        # Log prediction
        logger.info(f"Prediction made: {loan_status} (confidence: {confidence:.2f})")
        
        return {
            "Loan status": loan_status,
            "Confidence": f"{confidence:.2f}" if confidence > 0 else "N/A"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(applications: list[LoanApplication]):
    """
    Predict loan approval status for multiple applications.
    
    Args:
        applications: List of LoanApplication objects
        
    Returns:
        List of prediction results
    """
    try:
        results = []
        
        for i, application in enumerate(applications):
            # Convert application to DataFrame
            input_data = pd.DataFrame([application.dict()])
            
            # Scale numerical features
            input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Get prediction probability for confidence
            try:
                prediction_proba = model.predict_proba(input_data)
                confidence = float(np.max(prediction_proba))
            except AttributeError:
                confidence = 0.0
            
            # Convert prediction to human-readable format
            loan_status = "Approved" if prediction[0] == 1 else "Rejected"
            
            results.append({
                "application_id": f"APP_{i+1:03d}",
                "loan_status": loan_status,
                "confidence": f"{confidence:.2f}" if confidence > 0 else "N/A"
            })
        
        logger.info(f"Batch prediction completed for {len(applications)} applications")
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making batch prediction: {str(e)}"
        )

@app.get("/model/info", tags=["Model Information"])
async def get_model_info():
    """Get information about the loaded model."""
    try:
        model_info = {
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "feature_columns": [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
                "Loan_Amount_Term", "Credit_History", "Property_Area"
            ],
            "numerical_columns": numerical_columns,
            "scaler_type": type(scaler).__name__
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = model_info["feature_columns"]
            importances = model.feature_importances_
            model_info["feature_importance"] = dict(zip(feature_names, importances.tolist()))
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
