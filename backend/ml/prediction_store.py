"""
Persistent Prediction Store & Historical Analytics Engine.

Records prediction requests, model versions, predictions, latencies, feature hashes,
and request correlation IDs for auditability, model monitoring, and replay.
"""

import json
import hashlib
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.logger import logger
from backend.core.settings import settings

PREDICTION_STORE_PATH = Path(settings.LOG_DIR) / "prediction_store.jsonl"
_store_lock = threading.Lock()


class PredictionStore:
    """Manager for recording and analyzing historical predictions."""

    def __init__(self, log_path: Path = PREDICTION_STORE_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _hash_features(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def record_prediction(
        self,
        customer_id: str,
        prediction_result: Dict[str, Any],
        model_version: str = "v1.0.0",
        request_id: str = "-",
        latency_ms: float = 0.0,
        input_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record prediction record to Prediction Store."""
        feature_hash = self._hash_features(input_payload or {})

        record = {
            "prediction_id": f"pred_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}_{feature_hash[:6]}",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "customer_id": customer_id,
            "request_id": request_id,
            "model_version": model_version,
            "churn_probability": prediction_result.get("churn_probability"),
            "churn_prediction": prediction_result.get("churn_prediction"),
            "predicted_ltv": prediction_result.get("predicted_ltv"),
            "segment": prediction_result.get("segment"),
            "risk_level": prediction_result.get("risk_level"),
            "latency_ms": round(latency_ms, 3),
            "feature_hash": feature_hash,
        }

        with _store_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def get_history(
        self,
        customer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent prediction history records."""
        if not self.log_path.exists():
            return []

        records = []
        with _store_lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                if customer_id is None or item.get("customer_id") == customer_id:
                                    records.append(item)
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"PredictionStore: error reading history ({e})")
                return []

        return list(reversed(records[-limit:]))

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Return aggregated summary metrics over historical predictions."""
        history = self.get_history(limit=5000)
        if not history:
            return {
                "total_stored_predictions": 0,
                "avg_churn_probability": 0.0,
                "avg_ltv": 0.0,
                "avg_latency_ms": 0.0,
                "risk_breakdown": {},
            }

        total = len(history)
        churn_probs = [r["churn_probability"] for r in history if r.get("churn_probability") is not None]
        ltvs = [r["predicted_ltv"] for r in history if r.get("predicted_ltv") is not None]
        latencies = [r["latency_ms"] for r in history if r.get("latency_ms") is not None]

        risk_counts: Dict[str, int] = {}
        for r in history:
            rl = r.get("risk_level", "Unknown")
            risk_counts[rl] = risk_counts.get(rl, 0) + 1

        return {
            "total_stored_predictions": total,
            "avg_churn_probability": round(sum(churn_probs) / len(churn_probs), 4) if churn_probs else 0.0,
            "avg_ltv": round(sum(ltvs) / len(ltvs), 2) if ltvs else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "risk_breakdown": risk_counts,
        }


# Global PredictionStore instance
prediction_store = PredictionStore()
