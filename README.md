# Loan Approval Prediction System

A comprehensive machine learning project that predicts loan approval status based on applicant information using various algorithms and provides a REST API for real-time predictions.

## 🎯 Project Overview

This project implements a loan approval prediction system that analyzes applicant data to determine whether a loan should be approved or rejected. The system uses multiple machine learning algorithms to ensure accurate predictions and provides a user-friendly API for integration.

## 🚀 Features

- **Multiple ML Algorithms**: Comparison of Logistic Regression, SVM, Decision Trees, Random Forest, and Gradient Boosting
- **Hyperparameter Tuning**: Automated optimization using RandomizedSearchCV
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **REST API**: FastAPI-based API for real-time predictions
- **Model Persistence**: Save and load trained models
- **Professional Code Structure**: Modular design with separate components

## 📊 Dataset

The dataset contains information about loan applicants including:
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: Number of dependents (0, 1, 2, 3+)
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Yes/No
- **ApplicantIncome**: Income of the applicant
- **CoapplicantIncome**: Income of the co-applicant
- **LoanAmount**: Loan amount (in thousands)
- **Loan_Amount_Term**: Loan term (in months)
- **Credit_History**: Credit history (0: Bad, 1: Good)
- **Property_Area**: Urban/Semiurban/Rural

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
loan_prediction/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── model_training.py      # Model training and evaluation
│   └── utils.py              # Helper functions
├── models/
│   ├── loan_status_predictor.pkl  # Trained model
│   └── scaler.pkl                 # Fitted scaler
├── data/
│   └── loan_data.csv         # Dataset
├── notebooks/
│   └── loan_analysis.ipynb   # Jupyter notebook for analysis
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Usage

### Training the Model

1. **Run the Jupyter notebook** to train and evaluate models:
   ```bash
   jupyter notebook notebooks/loan_analysis.ipynb
   ```

2. **Or run the training script directly**:
   ```python
   from src.data_preprocessing import DataPreprocessor
   from src.model_training import ModelTrainer
   
   # Initialize components
   preprocessor = DataPreprocessor()
   trainer = ModelTrainer()
   
   # Preprocess data
   X, y = preprocessor.full_preprocessing_pipeline('data/loan_data.csv')
   
   # Train and compare models
   model_scores = trainer.compare_models(X, y)
   
   # Tune best model
   best_model = trainer.tune_hyperparameters('Random Forest', X, y)
   
   # Save model
   trainer.save_model(best_model, 'models/loan_status_predictor.pkl')
   ```

### Running the API

1. **Start the FastAPI server**:
   ```bash
   uvicorn app:app --reload
   ```

2. **Access the API documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Making Predictions

#### Using the API

Send a POST request to `http://localhost:8000/predict` with the following JSON payload:

```json
{
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
```

#### Using Python

```python
import requests
import json

# Sample data
data = {
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

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Loan Status: {result['Loan status']}")
```

## 📈 Model Performance

The project compares multiple machine learning algorithms:

| Algorithm | Cross-Validation Score | Accuracy |
|-----------|----------------------|----------|
| Random Forest | 0.82 | 0.85 |
| Gradient Boosting | 0.81 | 0.84 |
| Logistic Regression | 0.80 | 0.82 |
| SVM | 0.79 | 0.81 |
| Decision Tree | 0.75 | 0.78 |

*Note: Actual performance may vary based on hyperparameter tuning and data preprocessing.*

## 🔍 API Testing with Postman

### Test Cases

#### 1. Approved Loan Example
![Postman Test - Approved](images/postman_approved.png)

**Request:**
```json
{
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
```

**Response:**
```json
{
  "Loan status": "Approved"
}
```

#### 2. Rejected Loan Example
![Postman Test - Rejected](images/postman_rejected.png)

**Request:**
```json
{
  "Gender": 0,
  "Married": 0,
  "Dependents": 4,
  "Education": 0,
  "Self_Employed": 1,
  "ApplicantIncome": 1000,
  "CoapplicantIncome": 0,
  "LoanAmount": 500,
  "Loan_Amount_Term": 180,
  "Credit_History": 0,
  "Property_Area": 0
}
```

**Response:**
```json
{
  "Loan status": "Rejected"
}
```

### Screenshots Placeholder

*Add your Postman test screenshots here:*

1. **Approved Loan Test**: 
   - Screenshot showing successful prediction for an approved loan
   - Include request body, response, and status code

2. **Rejected Loan Test**:
   - Screenshot showing successful prediction for a rejected loan
   - Include request body, response, and status code

3. **API Documentation**:
   - Screenshot of Swagger UI documentation
   - Show available endpoints and request/response schemas

## 🧪 Testing

Run the test cases to verify the API functionality:

```bash
# Test approved loan case
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Gender": 1, "Married": 1, "Dependents": 0, "Education": 1, "Self_Employed": 0, "ApplicantIncome": 5000, "CoapplicantIncome": 2000, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1, "Property_Area": 2}'

# Test rejected loan case
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Gender": 0, "Married": 0, "Dependents": 4, "Education": 0, "Self_Employed": 1, "ApplicantIncome": 1000, "CoapplicantIncome": 0, "LoanAmount": 500, "Loan_Amount_Term": 180, "Credit_History": 0, "Property_Area": 0}'
```

## 🔧 Configuration

Edit `src/config.py` to modify:
- File paths
- Model parameters
- API settings
- Feature encodings

## 📋 Requirements

- Python 3.8+
- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- uvicorn

See `requirements.txt` for full dependencies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Dataset source: [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- Inspiration from various machine learning projects
- FastAPI documentation and community

---

*This project demonstrates proficiency in machine learning, data preprocessing, model evaluation, hyperparameter tuning, and API development. It's designed to showcase practical skills in building end-to-end ML solutions.*