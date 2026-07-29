"""
Unit tests for EnterpriseAlertManager.
"""

import pytest
from backend.core.alerts import EnterpriseAlertManager, SEVERITY_CRITICAL


def test_alert_manager_trigger_and_history(tmp_path):
    am = EnterpriseAlertManager()
    am.log_path = tmp_path / "test_alerts.jsonl"

    rec = am.trigger_alert(
        alert_type="TEST_ALERT",
        message="System alert integration test",
        severity=SEVERITY_CRITICAL,
        details={"component": "test"},
    )

    assert rec["alert_type"] == "TEST_ALERT"
    assert rec["severity"] == SEVERITY_CRITICAL
    assert "console" in rec["dispatched_channels"]

    history = am.get_alert_history(limit=10)
    assert len(history) == 1
    assert history[0]["alert_type"] == "TEST_ALERT"
