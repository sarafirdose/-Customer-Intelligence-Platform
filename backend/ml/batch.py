"""
Parallel Batch Prediction Engine.

Processes bulk customer datasets (CSV, Parquet, Excel) in parallel batches.
Calculates churn probability, remaining tenure, LTV, segments, and business actions.
Provides progress tracking and execution statistics.
"""

import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backend.core.logger import logger
from backend.services.predict_service import predict_churn


class BatchPredictionEngine:
    """High-performance batch prediction processor."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def load_dataset(self, file_source: Union[str, Path, bytes, io.BytesIO], file_type: str = "csv") -> pd.DataFrame:
        """Load customer dataset from CSV, Parquet, or Excel."""
        if isinstance(file_source, (str, Path)):
            path_str = str(file_source).lower()
            if path_str.endswith(".parquet") or file_type == "parquet":
                return pd.read_parquet(file_source)
            elif path_str.endswith((".xlsx", ".xls")) or file_type in ("excel", "xlsx"):
                return pd.read_excel(file_source)
            else:
                return pd.read_csv(file_source)
        else:
            # Handle in-memory bytes/stream
            if isinstance(file_source, bytes):
                buf = io.BytesIO(file_source)
            else:
                buf = file_source

            if file_type == "parquet":
                return pd.read_parquet(buf)
            elif file_type in ("excel", "xlsx"):
                return pd.read_excel(buf)
            else:
                return pd.read_csv(buf)

    def _score_single_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single customer row using predict_churn service."""
        customer_id = str(row_dict.get("customer_id", f"CUST_{uuid.uuid4().hex[:8]}"))
        try:
            res = predict_churn(row_dict)
            return {
                "customer_id": customer_id,
                "churn_probability": res.get("churn_probability", 0.0),
                "churn_prediction": res.get("churn_prediction", 0),
                "risk_level": res.get("risk_level", "Low"),
                "segment": res.get("segment", "Standard"),
                "predicted_ltv": res.get("predicted_ltv", 0.0),
                "intelligence_score": res.get("intelligence_score", 50.0),
                "action": res.get("recommendations", {}).get("action", "Monitor"),
                "status": "success",
                "error": None,
            }
        except Exception as e:
            return {
                "customer_id": customer_id,
                "churn_probability": None,
                "churn_prediction": None,
                "risk_level": "Unknown",
                "segment": "Unknown",
                "predicted_ltv": 0.0,
                "intelligence_score": 0.0,
                "action": "Investigate Data Error",
                "status": "error",
                "error": str(e),
            }

    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 50,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run parallel inference on DataFrame.

        Args:
            df: Customer dataframe.
            batch_size: Chunk size for thread mapping.
            progress_callback: Optional callable(completed, total).

        Returns:
            Dict with results dataframe, execution time, and summary statistics.
        """
        start_time = time.perf_counter()
        total_rows = len(df)
        records = df.to_dict(orient="records")

        results: List[Dict[str, Any]] = [None] * total_rows  # type: ignore[list-item]
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._score_single_row, rec): idx
                for idx, rec in enumerate(records)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {
                        "customer_id": str(records[idx].get("customer_id", f"ROW_{idx}")),
                        "status": "error",
                        "error": str(e),
                    }
                results[idx] = res
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_rows)

        results_df = pd.DataFrame(results)
        elapsed_sec = time.perf_counter() - start_time

        # Calculate statistics
        successful_df = results_df[results_df["status"] == "success"]
        successful_count = len(successful_df)
        error_count = total_rows - successful_count

        avg_churn = float(successful_df["churn_probability"].mean()) if successful_count > 0 else 0.0
        avg_ltv = float(successful_df["predicted_ltv"].mean()) if successful_count > 0 else 0.0
        high_risk_count = int((successful_df["risk_level"] == "High").sum()) if successful_count > 0 else 0

        summary = {
            "job_id": f"batch_{uuid.uuid4().hex[:8]}",
            "processed_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_records": total_rows,
            "successful_records": successful_count,
            "failed_records": error_count,
            "duration_seconds": round(elapsed_sec, 3),
            "throughput_rps": round(total_rows / elapsed_sec, 2) if elapsed_sec > 0 else 0.0,
            "avg_churn_probability": round(avg_churn, 4),
            "avg_predicted_ltv": round(avg_ltv, 2),
            "high_risk_customers": high_risk_count,
        }

        logger.info(
            f"BatchEngine: Processed {total_rows} records in {elapsed_sec:.2f}s "
            f"({summary['throughput_rps']} req/s)"
        )

        return {
            "summary": summary,
            "results_df": results_df,
        }


# Singleton instance
batch_engine = BatchPredictionEngine()
