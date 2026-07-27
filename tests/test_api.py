"""
Integration tests for FastAPI REST endpoints.

Validates route prefixes, request schema validation, prediction scoring, and health status.
"""

from fastapi import status
from fastapi.testclient import TestClient


def test_api_health_endpoint(client: TestClient) -> None:
    """
    Test that the detailed /health endpoint returns a 200 and healthy connection.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"
    assert payload["api"] == "running"


def test_api_predict_churn_endpoint(client: TestClient) -> None:
    """
    Test that the churn prediction endpoint validates schema and returns predictions.
    """
    customer_payload = {
        "customer_id": "7777-TEST",
        "tenure_months": 24,
        "monthly_charges": 55.40,
        "total_charges": 1329.60,
        "contract_type": "One year",
        "paperless_billing": "Yes",
        "internet_service": "DSL",
        "tech_support": "No",
    }

    response = client.post("/api/v1/predict", json=customer_payload)
    assert response.status_code == status.HTTP_200_OK

    payload = response.json()
    assert payload["customer_id"] == "7777-TEST"
    assert "churn_probability" in payload
    assert "is_churn" in payload
    assert "predicted_ltv" in payload
    assert payload["model_version"] == "1.0.0"


def test_api_predict_validation_error(client: TestClient) -> None:
    """
    Test that the churn prediction endpoint rejects incomplete payloads (HTTP 422).
    """
    invalid_payload = {
        "customer_id": "7777-TEST",
        # Missing tenure_months, charges etc.
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_api_explain_endpoint(client: TestClient) -> None:
    """
    Test that the model explanation endpoint returns feature attributions.
    """
    response = client.get("/api/v1/explain/7777-TEST")
    assert response.status_code == status.HTTP_200_OK

    payload = response.json()
    assert payload["customer_id"] == "7777-TEST"
    assert "base_value" in payload
    assert "attributions" in payload
    assert isinstance(payload["attributions"], dict)
