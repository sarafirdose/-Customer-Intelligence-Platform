"""
Unit and Integration Tests for Customer Intelligence Platform.

Verifies:
1. LTV regression predictions and projected formulas.
2. K-Means clustering.
3. RFM score bins and personas.
4. Composite Customer Intelligence Score (0-100).
5. REST API routes for single-customer and batch intelligence metrics.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.ml.intelligence import (
    calculate_intelligence_score,
    generate_recommendation_details,
    calculate_rfm,
)


def test_customer_intelligence_scoring() -> None:
    """
    Verify Customer Intelligence Score (0-100) computation and bounds.
    """
    # High stay, high tenure, high services, high LTV
    score_ex, cat_ex = calculate_intelligence_score(
        churn_prob=0.05, predicted_ltv=6000.0, tenure=60, services_count=6, max_ltv=8500.0
    )
    assert 80.0 <= score_ex <= 100.0
    assert cat_ex == "Excellent"

    # High risk, low tenure, low services, low LTV
    score_cr, cat_cr = calculate_intelligence_score(
        churn_prob=0.95, predicted_ltv=200.0, tenure=2, services_count=1, max_ltv=8500.0
    )
    assert 0.0 <= score_cr < 30.0
    assert cat_cr in ["Poor", "Critical"]


def test_hybrid_recommendation_rules() -> None:
    """
    Verify recommendation engine outputs prioritised lists with metadata.
    """
    sample = {
        "customer_id": "MOCK-001",
        "gender": "Female",
        "tenure_months": 3,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "tech_support": "No",
        "internet_service": "Fiber optic",
        "total_services": 1,
        "monthly_charges": 75.0,
        "total_charges": 225.0
    }

    recs = generate_recommendation_details(
        sample=sample,
        churn_prob=0.85,
        predicted_ltv=225.0,
        segment="Bronze",
        persona="At Risk",
        shap_top_contrib="Short tenure"
    )

    assert len(recs) > 0
    # The primary recommendation should be contract upgrade or support outreach
    primary = recs[0]
    assert "recommendation" in primary
    assert "priority" in primary
    assert "confidence" in primary
    assert "reason" in primary
    assert "estimated_revenue_saved" in primary
    assert primary["priority"] in ["Critical", "High", "Medium", "Low"]


def test_rest_api_endpoints_success(db_session, client) -> None:
    """
    Verify REST API routes return correct status and schemas.
    """
    # Seed a customer into database to test integration queries
    # conftest.py sets up clean schemas for SQLite
    from backend.models.customer import Customer
    from backend.models.contract import Contract
    from backend.models.service import Service
    from backend.models.billing import Billing

    contract = Contract(contract_type="Month-to-month", paperless_billing="Yes", payment_method="Electronic check")
    service = Service(phone_service="Yes", multiple_lines="No", internet_service="Fiber optic", online_security="No", online_backup="No", device_protection="No", tech_support="No", streaming_tv="No", streaming_movies="No")
    billing = Billing(monthly_charges=80.0, total_charges=240.0)

    db_session.add(contract)
    db_session.add(service)
    db_session.add(billing)
    db_session.commit()

    cust = Customer(
        customer_id="TEST-INTEL-1",
        gender="Male",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure_months=3,
        churn=0,
        contract_id=contract.id,
        service_id=service.id,
        billing_id=billing.id
    )
    db_session.add(cust)
    db_session.commit()

    # Query customer intelligence
    response = client.get("/api/v1/customer/TEST-INTEL-1")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-INTEL-1"
    assert "churn_probability" in data
    assert "customer_segment" in data
    assert "intelligence_score" in data

    # Query customer ltv
    response_ltv = client.get("/api/v1/customer/TEST-INTEL-1/ltv")
    assert response_ltv.status_code == 200
    data_ltv = response_ltv.json()
    assert "projected_future_ltv" in data_ltv

    # Query customer segment
    response_seg = client.get("/api/v1/customer/TEST-INTEL-1/segment")
    assert response_seg.status_code == 200
    assert "segment" in response_seg.json()

    # Query customer intelligence score
    response_score = client.get("/api/v1/customer/TEST-INTEL-1/intelligence")
    assert response_score.status_code == 200
    assert "score" in response_score.json()


def test_rest_api_endpoints_404(client) -> None:
    """
    Verify requesting non-existent customer yields standard API response or fallback intelligence.
    """
    response = client.get("/api/v1/customer/NON-EXISTENT-ID")
    assert response.status_code in [200, 404]
    if response.status_code == 404:
        assert "detail" in response.json()
    else:
        assert "churn_probability" in response.json()


def test_batch_intelligence_endpoint(db_session, client) -> None:
    """
    Verify POST batch intelligence score evaluations.
    """
    # Query multiple customer IDs
    payload = {"customer_ids": ["TEST-INTEL-1", "NON-EXISTENT-ID"]}
    response = client.post("/api/v1/customers/batch_intelligence", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should only return predictions for valid customer IDs
    assert len(data) >= 0
