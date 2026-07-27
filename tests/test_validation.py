"""
Unit tests for Pandera schema validations.

Validates raw and cleaned schemas with valid/invalid data configurations.
"""

import pandas as pd
import pytest
import pandera as pa
from backend.ml.validation import DataValidator


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    Expose a valid raw customer DataFrame representing Kaggle source files.
    """
    return pd.DataFrame(
        {
            "customerID": ["1111-MOCK", "2222-MOCK"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [45.85, 95.00],
            "TotalCharges": ["550.20", "2280.00"],
            "Churn": ["No", "Yes"],
        }
    )


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """
    Expose a valid cleaned customer DataFrame matching database types.
    """
    return pd.DataFrame(
        {
            "customer_id": ["1111-MOCK", "2222-MOCK"],
            "gender": ["Female", "Male"],
            "senior_citizen": [0, 1],
            "partner": ["Yes", "No"],
            "dependents": ["No", "Yes"],
            "tenure_months": [12, 24],
            "phone_service": ["Yes", "No"],
            "multiple_lines": ["No", "No phone service"],
            "internet_service": ["DSL", "Fiber optic"],
            "online_security": ["Yes", "No"],
            "online_backup": ["No", "Yes"],
            "device_protection": ["Yes", "No"],
            "tech_support": ["No", "Yes"],
            "streaming_tv": ["Yes", "No"],
            "streaming_movies": ["No", "Yes"],
            "contract_type": ["Month-to-month", "Two year"],
            "paperless_billing": ["Yes", "No"],
            "payment_method": ["Electronic check", "Mailed check"],
            "monthly_charges": [45.85, 95.00],
            "total_charges": [550.20, 2280.00],
            "churn": [0, 1],
        }
    )


def test_validator_raw_success(sample_raw_df: pd.DataFrame) -> None:
    """
    Test that a valid raw dataframe successfully passes raw schema validation.
    """
    validator = DataValidator()
    is_valid, report = validator.validate_raw(sample_raw_df)
    assert is_valid is True
    assert report["status"] == "success"


def test_validator_raw_failure(sample_raw_df: pd.DataFrame) -> None:
    """
    Test that raw validation flags invalid categories (e.g. invalid Gender).
    """
    bad_df = sample_raw_df.copy()
    bad_df.loc[0, "gender"] = "Unknown"  # Invalid gender category

    validator = DataValidator()
    is_valid, report = validator.validate_raw(bad_df)
    assert is_valid is False
    assert report["status"] == "failed"
    assert report["failures_count"] > 0


def test_validator_clean_success(sample_clean_df: pd.DataFrame) -> None:
    """
    Test that a valid cleaned dataframe successfully passes clean schema validation.
    """
    validator = DataValidator()
    is_valid, report = validator.validate_clean(sample_clean_df)
    assert is_valid is True
    assert report["status"] == "success"


def test_validator_clean_failure(sample_clean_df: pd.DataFrame) -> None:
    """
    Test that clean validation flags issues like duplicate IDs or invalid Churn flags.
    """
    bad_df = sample_clean_df.copy()
    bad_df.loc[0, "customer_id"] = "2222-MOCK"  # Create duplicate ID constraint violation

    validator = DataValidator()
    is_valid, report = validator.validate_clean(bad_df)
    assert is_valid is False
    assert report["status"] == "failed"
