"""
Unit tests for FeatureStore.
"""

import pytest
from backend.ml.feature_store import FeatureStore


def test_feature_store_validation():
    fs = FeatureStore(version="v1.0")

    valid_payload = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "contract_type": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "monthly_charges": 70.0,
        "total_charges": 840.0,
    }

    is_valid, errors = fs.validate_features(valid_payload)
    assert is_valid is True
    assert len(errors) == 0

    # Missing feature
    invalid_payload = dict(valid_payload)
    del invalid_payload["contract_type"]
    is_valid, errors = fs.validate_features(invalid_payload)
    assert is_valid is False
    assert any("contract_type" in err for err in errors)


def test_feature_store_defaults():
    fs = FeatureStore(version="v1.0")
    partial = {"tenure_months": 24}
    cleaned = fs.apply_defaults(partial)
    assert cleaned["tenure_months"] == 24
    assert cleaned["contract_type"] == "Month-to-month"
    assert cleaned["monthly_charges"] == 65.0
