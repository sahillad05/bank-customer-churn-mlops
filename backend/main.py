import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "frontend"

model = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    try:
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
        print("Model and preprocessor loaded successfully.")
    except Exception as e:
        print(f"Could not load model artifacts: {e}")
    yield
    # Clean up if needed
    model = None
    preprocessor = None

app = FastAPI(
    title="Bank Churn Predictor API", 
    version="1.0.0",
    lifespan=lifespan
)


# ──────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/metrics")
def get_metrics():
    """Return model evaluation metrics from the reports directory."""
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Run the DVC pipeline first.")
    with open(metrics_path, "r") as f:
        return JSONResponse(content=json.load(f))


@app.get("/confusion-matrix")
def get_confusion_matrix():
    """Return the confusion matrix PNG image."""
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    if not cm_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix not found. Run the DVC pipeline first.")
    return FileResponse(str(cm_path), media_type="image/png")


@app.post("/predict")
def predict(customer: CustomerData):
    """Run churn prediction for a single customer."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure the DVC pipeline has been run."
        )

    input_df = pd.DataFrame([{
        "CreditScore": customer.CreditScore,
        "Geography": customer.Geography,
        "Gender": customer.Gender,
        "Age": customer.Age,
        "Tenure": customer.Tenure,
        "Balance": customer.Balance,
        "NumOfProducts": customer.NumOfProducts,
        "HasCrCard": customer.HasCrCard,
        "IsActiveMember": customer.IsActiveMember,
        "EstimatedSalary": customer.EstimatedSalary,
    }])

    try:
        transformed = preprocessor.transform(input_df)
        prediction = int(model.predict(transformed)[0])
        proba = model.predict_proba(transformed)[0]
        churn_prob = float(proba[1])
        retention_prob = float(proba[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Determine risk level
    if churn_prob > 0.7:
        risk_level = "HIGH"
        recommendation = "Immediate retention action required. Consider personalized offers or direct outreach."
    elif churn_prob >= 0.4:
        risk_level = "MEDIUM"
        recommendation = "Moderate risk detected. Monitor engagement and consider proactive communication."
    else:
        risk_level = "LOW"
        recommendation = "Customer is likely to stay. Continue standard engagement practices."

    return JSONResponse(content={
        "prediction": prediction,
        "churn_probability": round(churn_prob * 100, 2),
        "retention_probability": round(retention_prob * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
    })


# ──────────────────────────────────────────────
# Serve static frontend — MUST be mounted last
# ──────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
