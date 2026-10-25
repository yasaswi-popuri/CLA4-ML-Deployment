
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML model serving using FastAPI",
    version="1.0.0"
)

# Load the existing trained pipeline
model = joblib.load("model.pkl")


# Define and validate incoming passenger data
class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str = Field(pattern="^(male|female)$")
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str | None = Field(default=None, pattern="^[CQS]$")


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert validated input into a DataFrame
    passenger_data = pd.DataFrame([passenger.model_dump()])

    # Generate prediction using the saved pipeline
    prediction = int(model.predict(passenger_data)[0])

    return {
        "predicted_survived": prediction,
        "result": (
            "Predicted to have survived"
            if prediction == 1
            else "Predicted not to have survived"
        )
    }
