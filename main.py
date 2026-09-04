from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")


class CustomerData(BaseModel):
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
    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str


@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.model_dump()])

    df = df.replace({
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
        "No internet service": 0,
        "No phone service": 0
    })

    df = pd.get_dummies(
        df,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod"
        ],
        drop_first=True
    )

    expected_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
        "InternetService_DSL",
        "InternetService_Fiber optic",
        "Contract_One year",
        "Contract_Two year",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check"
    ]

    df = df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }