"""
Unit tests for ContinuousRetrainingPipeline.
"""

import pytest
from backend.ml.retraining import ContinuousRetrainingPipeline


def test_retraining_pipeline_run():
    pipeline = ContinuousRetrainingPipeline()

    res = pipeline.run_retraining(
        dataset_path="reports/customer_intelligence.csv",
        auto_promote_production=False,
        trigger_type="test",
    )

    assert "run_id" in res
    assert res["status"] == "success"
    assert "new_metrics" in res
    assert "new_model_version" in res
    assert res["model_status"] == "staging"

    history = pipeline.get_history(limit=5)
    assert len(history) >= 1
