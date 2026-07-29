"""
Structured Audit Logging.

Writes one JSON object per line to logs/audit.jsonl.
Every audit event carries a correlation request_id from the ContextVar
set by the logging middleware, making full request traces greppable.

Event types:
    PREDICTION, BATCH_JOB, EXPORT, ERROR, USER_ACTION, SYSTEM
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.logger import logger, request_id_var
from backend.core.settings import settings

# Audit log location
AUDIT_LOG_PATH = Path(settings.LOG_DIR) / "audit.jsonl"

# Thread lock to prevent interleaved writes from concurrent requests
_write_lock = threading.Lock()

# Valid event types
EVENT_TYPES = {
    "PREDICTION",
    "BATCH_JOB",
    "EXPORT",
    "ERROR",
    "USER_ACTION",
    "SYSTEM",
    "REGISTRY",
    "DRIFT_CHECK",
    "SCHEDULER",
}


def log_audit_event(
    event_type: str,
    endpoint: str = "",
    customer_id: Optional[str] = None,
    model_version: Optional[str] = None,
    latency_ms: Optional[float] = None,
    result_summary: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a single audit event to logs/audit.jsonl.

    Args:
        event_type: One of the EVENT_TYPES constants.
        endpoint: API path or script name that triggered the event.
        customer_id: Customer account ID (if applicable).
        model_version: Active model version string.
        latency_ms: Processing time in milliseconds.
        result_summary: Dict summarising the outcome (scores, counts, etc.).
        error: Error message string (if event is an error).
        extra: Additional free-form metadata fields.
    """
    if event_type not in EVENT_TYPES:
        logger.warning(f"Audit: unknown event type '{event_type}' — logging as-is.")

    event: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": event_type,
        "request_id": request_id_var.get("-"),
        "endpoint": endpoint,
        "customer_id": customer_id,
        "model_version": model_version,
        "latency_ms": latency_ms,
        "result_summary": result_summary or {},
        "error": error,
    }
    if extra:
        event.update(extra)

    # Ensure directory exists
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _write_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_recent_audit_events(n: int = 50) -> list:
    """
    Read the last N audit events from the JSONL log.

    Args:
        n: Number of most recent events to return.

    Returns:
        List of event dicts, newest first.
    """
    if not AUDIT_LOG_PATH.exists():
        return []

    events = []
    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return []

    return list(reversed(events[-n:]))


def get_audit_stats() -> Dict[str, Any]:
    """Return basic statistics about the audit log."""
    if not AUDIT_LOG_PATH.exists():
        return {"total_events": 0, "by_type": {}, "log_size_kb": 0}

    events = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    by_type: Dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type", "UNKNOWN")
        by_type[et] = by_type.get(et, 0) + 1

    log_size_kb = round(AUDIT_LOG_PATH.stat().st_size / 1024, 2)

    return {
        "total_events": len(events),
        "by_type": by_type,
        "log_size_kb": log_size_kb,
        "log_path": str(AUDIT_LOG_PATH),
    }
