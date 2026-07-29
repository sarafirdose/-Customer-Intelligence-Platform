"""
Async Prediction Queue & Job Manager.

Manages background prediction tasks, supporting bulk enqueuing, progress tracking,
and status queries. Works with ThreadPool background worker or Redis Queue.
"""

import json
import os
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.core.logger import logger
from backend.services.predict_service import predict_churn


class PredictionQueueManager:
    """Task queue manager for async prediction workloads."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def enqueue_prediction(self, customer_data: Dict[str, Any]) -> str:
        """Enqueue a single asynchronous prediction task."""
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "completed_at": None,
                "customer_data": customer_data,
                "result": None,
                "error": None,
            }

        self._executor.submit(self._process_single_task, task_id, customer_data)
        return task_id

    def enqueue_batch(self, customers: List[Dict[str, Any]]) -> str:
        """Enqueue a bulk batch prediction job containing multiple records."""
        job_id = f"batch_job_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._tasks[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "completed_at": None,
                "total_records": len(customers),
                "processed_records": 0,
                "results": [],
                "error": None,
            }

        self._executor.submit(self._process_batch_task, job_id, customers)
        return job_id

    def _process_single_task(self, task_id: str, customer_data: Dict[str, Any]) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "processing"

        try:
            res = predict_churn(customer_data)
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]["status"] = "completed"
                    self._tasks[task_id]["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
                    self._tasks[task_id]["result"] = res
        except Exception as e:
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]["status"] = "failed"
                    self._tasks[task_id]["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
                    self._tasks[task_id]["error"] = str(e)

    def _process_batch_task(self, job_id: str, customers: List[Dict[str, Any]]) -> None:
        with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id]["status"] = "processing"

        results = []
        for idx, item in enumerate(customers):
            try:
                res = predict_churn(item)
                results.append({"index": idx, "status": "success", "prediction": res})
            except Exception as e:
                results.append({"index": idx, "status": "error", "error": str(e)})

            with self._lock:
                if job_id in self._tasks:
                    self._tasks[job_id]["processed_records"] = idx + 1

        with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id]["status"] = "completed"
                self._tasks[job_id]["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
                self._tasks[job_id]["results"] = results

    def get_task_status(self, task_or_job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an async prediction task or job."""
        with self._lock:
            return self._tasks.get(task_or_job_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue metrics and status totals."""
        with self._lock:
            total = len(self._tasks)
            queued = sum(1 for t in self._tasks.values() if t["status"] == "queued")
            processing = sum(1 for t in self._tasks.values() if t["status"] == "processing")
            completed = sum(1 for t in self._tasks.values() if t["status"] == "completed")
            failed = sum(1 for t in self._tasks.values() if t["status"] == "failed")

            return {
                "total_tasks": total,
                "queued": queued,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "max_workers": self.max_workers,
            }


# Singleton instance
queue_manager = PredictionQueueManager()
