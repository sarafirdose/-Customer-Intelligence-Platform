"""
Unit tests for PredictionStore.
"""

import pytest
from backend.ml.prediction_store import PredictionStore


def test_prediction_store_record_and_get(tmp_path):
    log_file = tmp_path / "test_prediction_store.jsonl"
    ps = PredictionStore(log_path=log_file)

    res = {
        "churn_probability": 0.35,
        "churn_prediction": 0,
        "predicted_ltv": 1500.0,
        "segment": "High Value",
        "risk_level": "Low",
    }

    record = ps.record_prediction(
        customer_id="C100",
        prediction_result=res,
        model_version="v1.0.0",
        request_id="req_123",
        latency_ms=15.2,
        input_payload={"tenure_months": 24},
    )

    assert record["customer_id"] == "C100"
    assert record["churn_probability"] == 0.35
    assert record["request_id"] == "req_123"

    history = ps.get_history(customer_id="C100", limit=10)
    assert len(history) == 1
    assert history[0]["customer_id"] == "C100"

    summary = ps.get_analytics_summary()
    assert summary["total_stored_predictions"] == 1
    assert summary["avg_churn_probability"] == 0.35
