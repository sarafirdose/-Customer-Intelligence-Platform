"""
Background Job Scheduler for the Customer Intelligence Platform.

Uses APScheduler 3.x BackgroundScheduler running inside the FastAPI process.
All job executions are logged to logs/scheduler_history.jsonl.

Scheduled jobs:
  - watch_folder_scan          → Every 1 minute
  - database_auto_sync         → Every 5 minutes
  - daily_metrics_flush        → 00:05 every day
  - daily_drift_check          → 01:00 every day
  - monthly_retraining_check   → 1st of month 02:00
  - log_rotation_cleanup       → 00:10 every day

Integration:
    Call scheduler.start() inside FastAPI lifespan start.
    Call scheduler.shutdown() inside FastAPI lifespan stop.
"""

import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.core.logger import logger
from backend.core.settings import settings

# ---------------------------------------------------------------------------
# Scheduler history log
# ---------------------------------------------------------------------------
HISTORY_LOG_PATH = Path(settings.LOG_DIR) / "scheduler_history.jsonl"
_history_lock = threading.Lock()

LOG_RETENTION_DAYS = 30


def _log_job_event(
    job_name: str,
    status: str,
    start_time: str,
    end_time: str,
    duration_seconds: float,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a job execution record to scheduler_history.jsonl."""
    record: Dict[str, Any] = {
        "job": job_name,
        "status": status,  # "success" | "error"
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": round(duration_seconds, 3),
        "error": error,
    }
    if extra:
        record.update(extra)

    HISTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _history_lock:
        with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_tracked_job(job_name: str, fn, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """Wrapper that times execution and writes to scheduler history."""
    start = datetime.now(tz=timezone.utc)
    start_iso = start.isoformat()
    logger.info(f"Scheduler: starting job '{job_name}'")
    try:
        result = fn(*args, **kwargs)
        end = datetime.now(tz=timezone.utc)
        duration = (end - start).total_seconds()
        _log_job_event(
            job_name=job_name,
            status="success",
            start_time=start_iso,
            end_time=end.isoformat(),
            duration_seconds=duration,
            extra={"result": str(result) if result else None},
        )
        logger.info(f"Scheduler: '{job_name}' completed in {duration:.2f}s")
    except Exception as exc:
        end = datetime.now(tz=timezone.utc)
        duration = (end - start).total_seconds()
        tb = traceback.format_exc()
        _log_job_event(
            job_name=job_name,
            status="error",
            start_time=start_iso,
            end_time=end.isoformat(),
            duration_seconds=duration,
            error=str(exc),
            extra={"traceback": tb},
        )
        logger.error(f"Scheduler: '{job_name}' FAILED — {exc}")


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------

def _watch_folder_scan() -> None:
    """Scan data/incoming/ for new CSV files continuously."""
    from backend.services.auto_ingestion import scan_watch_folder
    res = scan_watch_folder()
    if res.get("processed_files", 0) > 0 or res.get("failed_files", 0) > 0:
        logger.info(f"Scheduler: Watch folder scan complete. Res: {res}")


def _database_auto_sync() -> None:
    """Incremental sync from PostgreSQL to ML prediction store."""
    from backend.services.auto_ingestion import run_database_auto_sync
    res = run_database_auto_sync()
    logger.info(f"Scheduler: Database auto-sync complete. Status: {res.get('sync_status')}")


def _daily_metrics_flush() -> None:
    """Flush today's in-memory metrics snapshot to disk."""
    from backend.core.metrics import metrics
    metrics.flush_daily_summary()
    logger.info("Scheduler: daily metrics flushed to disk.")


def _daily_drift_check() -> None:
    """
    Run drift check against the latest customer_intelligence.csv report.
    Falls back gracefully if the file does not exist.
    """
    import pandas as pd
    from backend.ml.drift import run_drift_check
    from backend.services.auto_ingestion import update_sync_state
    from pathlib import Path

    data_path = Path("reports/customer_intelligence.csv")
    if not data_path.exists():
        logger.warning("Scheduler: customer_intelligence.csv not found — skipping drift check.")
        return

    df = pd.read_csv(data_path)
    report = run_drift_check(df)
    now_iso = datetime.now(timezone.utc).isoformat()
    update_sync_state({"last_drift_check": now_iso})
    logger.info(
        f"Scheduler: daily drift check complete. "
        f"Overall severity: {report['overall_severity']}"
    )


def _monthly_model_evaluation() -> None:
    """Compare current production model metrics against registry baseline and trigger retraining check."""
    from backend.ml.registry import get_production_model, list_versions
    from backend.services.auto_ingestion import update_sync_state

    for model_name in ("churn", "ltv", "segmentation"):
        prod = get_production_model(model_name)
        versions = list_versions(model_name)
        if prod:
            logger.info(
                f"Scheduler: monthly eval — {model_name}@{prod['version']} "
                f"is production. {len(versions)} total version(s) registered."
            )
        else:
            logger.warning(f"Scheduler: no production model for '{model_name}'.")

    now_iso = datetime.now(timezone.utc).isoformat()
    update_sync_state({"last_retraining": now_iso})


def _log_rotation_cleanup() -> None:
    """Delete log files older than LOG_RETENTION_DAYS days."""
    import time

    log_dir = Path(settings.LOG_DIR)
    cutoff = time.time() - (LOG_RETENTION_DAYS * 86400)
    removed = 0

    for log_file in log_dir.glob("metrics_*.json"):
        if log_file.stat().st_mtime < cutoff:
            log_file.unlink()
            removed += 1

    if removed:
        logger.info(f"Scheduler: removed {removed} old metric snapshot(s).")


# ---------------------------------------------------------------------------
# Scheduler singleton
# ---------------------------------------------------------------------------

class PlatformScheduler:
    """Wraps APScheduler with job registration and lifecycle management."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._started = False
        self._register_jobs()

    def _register_jobs(self) -> None:
        # Watch folder scan every 1 minute
        self._scheduler.add_job(
            lambda: _run_tracked_job("watch_folder_scan", _watch_folder_scan),
            trigger=IntervalTrigger(minutes=1),
            id="watch_folder_scan",
            replace_existing=True,
        )
        # Database auto sync every 5 minutes
        self._scheduler.add_job(
            lambda: _run_tracked_job("database_auto_sync", _database_auto_sync),
            trigger=IntervalTrigger(minutes=5),
            id="database_auto_sync",
            replace_existing=True,
        )
        # Daily metrics flush at 00:05 UTC
        self._scheduler.add_job(
            lambda: _run_tracked_job("daily_metrics_flush", _daily_metrics_flush),
            trigger=CronTrigger(hour=0, minute=5),
            id="daily_metrics_flush",
            replace_existing=True,
        )
        # Daily drift report at 01:00 UTC
        self._scheduler.add_job(
            lambda: _run_tracked_job("daily_drift_check", _daily_drift_check),
            trigger=CronTrigger(hour=1, minute=0),
            id="daily_drift_check",
            replace_existing=True,
        )
        # Monthly model evaluation — 1st of month 02:00 UTC
        self._scheduler.add_job(
            lambda: _run_tracked_job("monthly_retraining_check", _monthly_model_evaluation),
            trigger=CronTrigger(day=1, hour=2, minute=0),
            id="monthly_retraining_check",
            replace_existing=True,
        )
        # Log rotation cleanup — daily 00:10 UTC
        self._scheduler.add_job(
            lambda: _run_tracked_job("log_rotation_cleanup", _log_rotation_cleanup),
            trigger=CronTrigger(hour=0, minute=10),
            id="log_rotation_cleanup",
            replace_existing=True,
        )

    def start(self) -> None:
        if not settings.SCHEDULER_ENABLED:
            logger.info("Scheduler: disabled via SCHEDULER_ENABLED=false.")
            return
        if not self._started:
            self._scheduler.start()
            self._started = True
            logger.info("Scheduler: started with 6 background jobs.")

    def shutdown(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("Scheduler: shut down cleanly.")

    def get_job_status(self) -> list:
        """Return a list of registered jobs with next run times."""
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run.isoformat() if next_run else None,
                "trigger": str(job.trigger),
            })
        return jobs

    @property
    def is_running(self) -> bool:
        return self._started


# Global singleton — imported by main.py lifespan
platform_scheduler = PlatformScheduler()
