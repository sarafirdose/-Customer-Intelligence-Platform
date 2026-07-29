"""
Unit tests for DeploymentManager, QueueManager, and ExplainabilityService.
"""

import pytest
from backend.ml.deployment_manager import DeploymentManager
from backend.ml.explain import ExplainabilityService
from backend.workers.queue_manager import PredictionQueueManager


def test_deployment_manager():
    dm = DeploymentManager()
    status = dm.get_deployment_status()
    assert "environment" in status
    assert "production_model_version" in status

    # Update Canary
    status = dm.update_canary("v1.1.0", 25)
    assert status["canary"]["traffic_percentage"] == 25
    assert status["canary"]["target_version"] == "v1.1.0"

    # Switch Blue/Green
    initial_env = status["environment"]
    switched = dm.switch_bluegreen()
    assert switched["environment"] != initial_env


def test_explainability_service():
    service = ExplainabilityService()

    global_imp = service.get_global_importance()
    assert len(global_imp) > 0
    assert "feature" in global_imp[0]

    explanation = service.explain_customer("C001", {"contract_type": "Month-to-month", "tenure_months": 5})
    assert explanation["customer_id"] == "C001"
    assert "waterfall_data" in explanation
    assert "force_plot_data" in explanation
    assert len(explanation["top_positive_drivers"]) > 0


def test_prediction_queue_manager():
    qm = PredictionQueueManager(max_workers=2)

    task_id = qm.enqueue_prediction({
        "gender": "Female",
        "tenure_months": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0,
    })

    assert task_id.startswith("task_")

    status = qm.get_task_status(task_id)
    assert status is not None
    assert status["status"] in ("queued", "processing", "completed")

    stats = qm.get_stats()
    assert stats["total_tasks"] >= 1
