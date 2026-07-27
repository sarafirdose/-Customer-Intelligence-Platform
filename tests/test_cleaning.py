"""
Unit tests for data cleaning pipelines.

Validates whitespace stripping, column naming, typecasts, and duplicate drops.
"""

import numpy as np
import pandas as pd
from backend.ml.cleaning import DataCleaner


def test_cleaner_renaming_and_types() -> None:
    """
    Test that cleaner standardizes column headers and coerces datatypes.
    """
    raw_data = pd.DataFrame(
        {
            "customerID": ["  1234-CUST  "],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [10],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Mailed check"],
            "MonthlyCharges": [29.95],
            "TotalCharges": [" 299.50 "],
            "Churn": ["No"],
        }
    )

    cleaner = DataCleaner()
    cleaned_df, summary = cleaner.clean(raw_data)

    # Validate column renaming
    assert "customer_id" in cleaned_df.columns
    assert "tenure_months" in cleaned_df.columns
    assert "total_charges" in cleaned_df.columns

    # Validate whitespace trim on IDs
    assert cleaned_df.loc[0, "customer_id"] == "1234-CUST"

    # Validate type casting
    assert isinstance(cleaned_df.loc[0, "total_charges"], float)
    assert cleaned_df.loc[0, "total_charges"] == 299.50
    assert cleaned_df.loc[0, "churn"] == 0


def test_cleaner_imputes_missing_total_charges() -> None:
    """
    Test that empty spaces in TotalCharges are handled (set to 0.0 for tenure = 0).
    """
    raw_data = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [0, 5],  # 0 tenure vs 5 tenure
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Mailed check", "Mailed check"],
            "MonthlyCharges": [20.0, 50.0],
            "TotalCharges": [" ", " "],  # Empty spaces in both
            "Churn": ["No", "No"],
        }
    )

    cleaner = DataCleaner()
    cleaned_df, summary = cleaner.clean(raw_data)

    # C1 (tenure=0) total charges should be 0.0
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C1", "total_charges"].values[0] == 0.0

    # C2 (tenure=5) total charges should fallback to monthly charges (50.0)
    assert cleaned_df.loc[cleaned_df["customer_id"] == "C2", "total_charges"].values[0] == 50.0


def test_cleaner_drops_duplicates() -> None:
    """
    Test that duplicate customerIDs are removed.
    """
    raw_data = pd.DataFrame(
        {
            "customerID": ["DUP-1", "DUP-1"],
            "gender": ["Male", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "Yes"],
            "Dependents": ["No", "No"],
            "tenure": [12, 14],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Mailed check", "Mailed check"],
            "MonthlyCharges": [25.0, 25.0],
            "TotalCharges": ["300.0", "350.0"],
            "Churn": ["No", "No"],
        }
    )

    cleaner = DataCleaner()
    cleaned_df, summary = cleaner.clean(raw_data)

    # Should retain only 1 row (the first one)
    assert len(cleaned_df) == 1
    assert summary["duplicates_removed"] == 1
    assert cleaned_df.loc[0, "total_charges"] == 300.0
