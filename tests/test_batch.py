"""
Unit tests for BatchPredictionEngine.
"""

import pandas as pd
import pytest
from backend.ml.batch import BatchPredictionEngine


def test_batch_prediction_engine():
    engine = BatchPredictionEngine(max_workers=2)

    df = pd.DataFrame([
        {
            "customer_id": "C001",
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure_months": 12,
            "contract_type": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 70.0,
            "total_charges": 840.0,
        },
        {
            "customer_id": "C002",
            "gender": "Male",
            "senior_citizen": 1,
            "partner": "No",
            "dependents": "No",
            "tenure_months": 48,
            "contract_type": "Two year",
            "paperless_billing": "No",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 50.0,
            "total_charges": 2400.0,
        },
    ])

    res = engine.process_batch(df)
    assert "summary" in res
    assert "results_df" in res

    summary = res["summary"]
    assert summary["total_records"] == 2
    assert summary["successful_records"] == 2
    assert summary["failed_records"] == 0

    results_df = res["results_df"]
    assert len(results_df) == 2
    assert "churn_probability" in results_df.columns
    assert "risk_level" in results_df.columns
