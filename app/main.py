from fastapi import FastAPI 
from pydantic import BaseModel
import joblib 
import pandas as pd 
from pathlib import Path

class Customer(BaseModel): 
    gender: str 
    SeniorCitizen: int 
    Partner: str 
    Dependents: str 
    tenure: int 
    PhoneService: str 
    MultipleLines: str 
    InternetService: str 
    OnlineSecurity: str 
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingTV: str 
    StreamingMovies: str 
    Contract: str 
    PaperlessBilling: str 
    PaymentMethod: str 
    MonthlyCharges: float 
    TotalCharges: float 

app = FastAPI() 

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lr-churn-model.joblib"
model = joblib.load(MODEL_PATH)

@app.get("/")
def root(): 
    return {"message":"Churn API is running"} 


@app.post("/predict")
def predict(customer: Customer): 
    c = customer.model_dump() # convert to python dic 
    cdf = pd.DataFrame(data=[c]) # convert dictionary to pandas df 
    prediction = model.predict(cdf) # returns a ndarray  
    churn_prob = model.predict_proba(cdf)
    return {
        "prediction": prediction.item(0),
        "churn-probability": round(churn_prob.item(1), 2)
    }