import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
 
app = FastAPI(
    title="Loan Prediction API",
    description="This page shows loan prediction.",
    version="1.0"
)

 
model = joblib.load("./model/loan_default.pkl")
 
# Ensure diagnostic sample matches training feature expectations
expected_features = [
    "Current Loan Amount", "Term", "Credit Score", "Annual Income",
    "Years in current job", "Home Ownership", "Purpose", "Monthly Debt",
    "Years of Credit History", "Months since last delinquent",
    "Number of Open Accounts", "Number of Credit Problems",
    "Current Credit Balance", "Maximum Open Credit", "Bankruptcies", "Tax Liens"
]

 
# REMOVED: Duplicate Pydantic imports that were overwriting configs
class LoanInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
 
    current_loan_amount: float = Field(alias="Current Loan Amount")
    term: float = Field(alias="Term")
    credit_score: float = Field(alias="Credit Score")
    annual_income: float = Field(alias="Annual Income")
    years_in_current_job: float = Field(alias="Years in current job")
    home_ownership: float = Field(alias="Home Ownership")
    purpose: float = Field(alias="Purpose")
    monthly_debt: float = Field(alias="Monthly Debt")
    years_of_credit_history: float = Field(alias="Years of Credit History")
    months_since_last_delinquent: float = Field(alias="Months since last delinquent")
    number_of_open_accounts: float = Field(alias="Number of Open Accounts")
    number_of_credit_problems: float = Field(alias="Number of Credit Problems")
    current_credit_balance: float = Field(alias="Current Credit Balance")
    maximum_open_credit: float = Field(alias="Maximum Open Credit")
    bankruptcies: float = Field(alias="Bankruptcies")
    tax_liens: float = Field(alias="Tax Liens")
 
@app.get("/")
def home():
    return {
        "message": "Loan Prediction API"
    }
 
@app.post("/predict")
def predict(data: LoanInput):
    # FIX 1: Explicitly pass by_alias=True to generate fields matching the model names
    raw_data = data.model_dump(by_alias=True)
    input_data = pd.DataFrame([raw_data])
 
    # FIX 2: Enforce correct column order before model prediction
    input_data = input_data[expected_features]
 
    prediction = model.predict(input_data)
 
    return {
        "prediction": float(prediction[0])
    }
 