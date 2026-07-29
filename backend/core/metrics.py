"""
Thread-Safe Prediction & API Metrics Collector.

Maintains in-memory counters updated on every request via MetricsMiddleware.
Flushes a daily summary snapshot to logs/metrics_YYYY-MM-DD.json.

Usage:
    from backend.core.metrics import metrics
    metrics.record_request("/api/v1/predict", latency_ms=12.3, error=False)
"""

import json
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from backend.core.settings import settings


class MetricsCollector:
    """
    Centralized application metrics store.

    All public methods are thread-safe via a reentrant lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_time = datetime.now(tz=timezone.utc)

        # Request counters
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._requests_per_endpoint: Dict[str, int] = defaultdict(int)
        self._errors_per_endpoint: Dict[str, int] = defaultdict(int)

        # Latency tracking (rolling window of last 1000 measurements)
        self._latencies_ms: deque = deque(maxlen=1000)
        self._latencies_per_endpoint: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

        # Prediction tracking
        self._prediction_count: int = 0
        self._churn_prob_sum: float = 0.0
        self._ltv_sum: float = 0.0

        # Batch tracking
        self._batch_job_count: int = 0
        self._batch_total_records: int = 0

        # Log directory
        self._log_dir = Path(settings.LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Recording methods                                                    #
    # ------------------------------------------------------------------ #

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a single API request's outcome."""
        with self._lock:
            self._total_requests += 1
            self._requests_per_endpoint[endpoint] += 1
            self._latencies_ms.append(latency_ms)
            self._latencies_per_endpoint[endpoint].append(latency_ms)
            if error:
                self._total_errors += 1
                self._errors_per_endpoint[endpoint] += 1

    def record_prediction(
        self,
        churn_probability: Optional[float] = None,
        predicted_ltv: Optional[float] = None,
    ) -> None:
        """Record a churn/LTV prediction outcome."""
        with self._lock:
            self._prediction_count += 1
            if churn_probability is not None:
                self._churn_prob_sum += churn_probability
            if predicted_ltv is not None:
                self._ltv_sum += predicted_ltv

    def record_batch_job(self, record_count: int) -> None:
        """Record completion of a batch scoring job."""
        with self._lock:
            self._batch_job_count += 1
            self._batch_total_records += record_count

    # ------------------------------------------------------------------ #
    # Snapshot                                                             #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time metrics snapshot."""
        with self._lock:
            lats = list(self._latencies_ms)
            uptime = (datetime.now(tz=timezone.utc) - self._start_time).total_seconds()

            proc = psutil.Process(os.getpid())
            mem_mb = round(proc.memory_info().rss / 1024 / 1024, 2)
            cpu_pct = psutil.cpu_percent(interval=None)

            def _pct(n: int) -> float:
                sorted_lats = sorted(lats)
                if not sorted_lats:
                    return 0.0
                idx = max(0, int(len(sorted_lats) * n / 100) - 1)
                return round(sorted_lats[idx], 3)

            avg_churn = (
                round(self._churn_prob_sum / self._prediction_count, 4)
                if self._prediction_count > 0 else 0.0
            )
            avg_ltv = (
                round(self._ltv_sum / self._prediction_count, 2)
                if self._prediction_count > 0 else 0.0
            )
            error_rate = (
                round(self._total_errors / self._total_requests, 4)
                if self._total_requests > 0 else 0.0
            )

            return {
                "collected_at": datetime.now(tz=timezone.utc).isoformat(),
                "uptime_seconds": round(uptime, 1),
                "cpu_percent": cpu_pct,
                "memory_mb": mem_mb,
                "requests": {
                    "total": self._total_requests,
                    "errors": self._total_errors,
                    "error_rate": error_rate,
                    "per_endpoint": dict(self._requests_per_endpoint),
                },
                "latency_ms": {
                    "avg": round(sum(lats) / len(lats), 3) if lats else 0.0,
                    "p50": _pct(50),
                    "p95": _pct(95),
                    "p99": _pct(99),
                },
                "predictions": {
                    "count": self._prediction_count,
                    "avg_churn_probability": avg_churn,
                    "avg_ltv": avg_ltv,
                },
                "batch_jobs": {
                    "count": self._batch_job_count,
                    "total_records": self._batch_total_records,
                },
            }

    def flush_daily_summary(self) -> None:
        """Persist today's metrics snapshot to a dated JSON file."""
        snap = self.snapshot()
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        out_path = self._log_dir / f"metrics_{today}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)

    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        with self._lock:
            self.__init__()  # type: ignore[misc]


# Global singleton
metrics = MetricsCollector()
