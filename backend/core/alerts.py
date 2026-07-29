"""
Enterprise Alert Manager.

Dispatches alerts across Slack, Teams, Email, and Structured Logs for critical platform events:
  - Critical Drift
  - API Down / Unhealthy
  - High Latency (>500ms)
  - Scheduler Failure
  - Model Failure
  - High Error Rate (>5%)
  - Low Model Accuracy
"""

import json
import os
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.logger import logger
from backend.core.settings import settings

ALERT_HISTORY_PATH = Path(settings.LOG_DIR) / "alerts_history.jsonl"
_alert_lock = threading.Lock()

# Alert Severity Levels
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"


class EnterpriseAlertManager:
    """Centralized Alert Dispatcher."""

    def __init__(self) -> None:
        self.log_path = ALERT_HISTORY_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.slack_url = os.getenv("SLACK_WEBHOOK_URL") or getattr(settings, "SLACK_WEBHOOK_URL", "")
        self.teams_url = os.getenv("TEAMS_WEBHOOK_URL") or getattr(settings, "TEAMS_WEBHOOK_URL", "")

    def trigger_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = SEVERITY_WARNING,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger an alert across configured channels.

        Args:
            alert_type: Event type e.g. 'CRITICAL_DRIFT', 'HIGH_LATENCY', 'MODEL_FAILURE'
            message: Human readable notification description.
            severity: 'INFO', 'WARNING', or 'CRITICAL'
            details: Extra metadata dictionary.

        Returns:
            Alert record dictionary.
        """
        alert_id = f"alert_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"
        record = {
            "alert_id": alert_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "dispatched_channels": [],
        }

        # 1. Structured Log & Console
        log_msg = f"ALERT [{severity}] [{alert_type}] {message}"
        if severity == SEVERITY_CRITICAL:
            logger.error(log_msg)
        else:
            logger.warning(log_msg)
        record["dispatched_channels"].append("console")

        # 2. Slack Webhook (if configured)
        if self.slack_url:
            try:
                payload = json.dumps({"text": f"🚨 *{alert_type}* [{severity}]\n{message}"}).encode("utf-8")
                req = urllib.request.Request(self.slack_url, data=payload, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3.0)
                record["dispatched_channels"].append("slack")
            except Exception as e:
                logger.warning(f"AlertManager: Slack webhook failed ({e})")

        # 3. Teams Webhook (if configured)
        if self.teams_url:
            try:
                payload = json.dumps({"text": f"🚨 **{alert_type}** [{severity}]\n{message}"}).encode("utf-8")
                req = urllib.request.Request(self.teams_url, data=payload, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3.0)
                record["dispatched_channels"].append("teams")
            except Exception as e:
                logger.warning(f"AlertManager: Teams webhook failed ({e})")

        # 4. Save to history log
        with _alert_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent alert history."""
        if not self.log_path.exists():
            return []
        records = []
        with _alert_lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass
            except Exception:
                return []
        return list(reversed(records[-limit:]))


# Global AlertManager instance
alert_manager = EnterpriseAlertManager()
