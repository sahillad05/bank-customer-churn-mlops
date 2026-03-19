import pytest
from fastapi.testclient import TestClient
from backend.main import app

def test_read_metrics():
    """Verify that the /metrics endpoint is reachable and returns valid JSON."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "accuracy" in response.json()

def test_read_confusion_matrix():
    """Verify that the /confusion-matrix endpoint returns a PNG image."""
    with TestClient(app) as client:
        response = client.get("/confusion-matrix")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

def test_predict_endpoint():
    """Verify that the /predict endpoint handles a valid customer profile."""
    test_data = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data

def test_invalid_predict_data():
    """Verify that the /predict endpoint returns 422 for missing data."""
    test_data = {"Age": 40} # Incomplete data
    with TestClient(app) as client:
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422
