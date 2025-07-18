# Setup Instructions

## Quick Start Guide

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the System

```bash
# Run system tests
python test_system.py
```

### 3. Run the API

```bash
# Start the FastAPI server
uvicorn app:app --reload
```

### 4. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

#### Sample API Request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

### 5. Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/loan_analysis.ipynb
```

## File Structure

```
loan_prediction/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── data_preprocessing.py     # Data preprocessing
│   ├── model_training.py         # Model training
│   └── utils.py                  # Utilities
├── data/                         # Dataset
│   └── loan_data.csv
├── models/                       # Trained models
│   ├── loan_status_predictor.pkl
│   └── scaler.pkl
├── notebooks/                    # Jupyter notebooks
│   ├── loan_analysis.ipynb       # Main analysis notebook
│   └── old_loan_approval_prediction.ipynb
├── images/                       # Screenshots and images
├── app.py                        # FastAPI application
├── test_system.py                # System tests
├── requirements.txt              # Dependencies
├── setup.md                      # This file
└── README.md                     # Project documentation
```

## Troubleshooting

### Common Issues:

1. **Module not found errors**: Make sure you're in the project root directory
2. **Model/scaler not found**: Ensure files are in the `models/` directory
3. **API not starting**: Check if port 8000 is available

### If you need to retrain the model:

1. Run the Jupyter notebook: `notebooks/loan_analysis.ipynb`
2. Execute all cells to retrain and save the model
3. Test the system: `python test_system.py`

## Next Steps

1. **Add your Postman screenshots** to the `images/` folder
2. **Update the README.md** with your personal information
3. **Test the API thoroughly** with different inputs
4. **Deploy to cloud** (Heroku, AWS, etc.) for portfolio showcase
