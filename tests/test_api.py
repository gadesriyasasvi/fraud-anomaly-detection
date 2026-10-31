from fastapi.testclient import TestClient
from fraud_engine.api import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_schema_validation():
    response = TestClient(app).post("/predict", json={"transaction_id": "TX1", "amount": 50})
    assert response.status_code == 422
