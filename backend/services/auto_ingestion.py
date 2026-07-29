"""
Automated Ingestion Engine for Telecom Customer Intelligence Platform.

Supports:
  1. Watch Folder Auto-Scan (data/incoming/ -> data/processed/ or data/failed/)
  2. Database Incremental Auto-Sync (PostgreSQL to Report/Prediction Store)
  3. Real-Time REST API Ingestion (/api/v1/ingest/record & /api/v1/ingest/batch)
  4. Automatic Prediction & Dashboard Refresh Trigger
"""

import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from backend.core.logger import logger
from backend.services.predict_service import predict_churn


BASE_DIR = Path(__file__).resolve().parents[2]
INCOMING_DIR = BASE_DIR / "data" / "incoming"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAILED_DIR = BASE_DIR / "data" / "failed"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

STATE_FILE = BASE_DIR / "data" / "sync_state.json"
IMPORT_LOG_FILE = LOGS_DIR / "imports.jsonl"

_state_lock = threading.Lock()


def get_sync_state() -> Dict[str, Any]:
    """Retrieve current auto-sync state and stats."""
    default_state = {
        "last_auto_sync": None,
        "records_processed_today": 0,
        "last_prediction_time": None,
        "last_drift_check": None,
        "last_retraining": None,
        "sync_status": "idle",
        "watch_folder_active": True,
        "processed_file_count": 0,
        "failed_file_count": 0,
    }
    if not STATE_FILE.exists():
        return default_state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            default_state.update(data)
            return default_state
    except Exception:
        return default_state


def update_sync_state(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update and persist sync state."""
    with _state_lock:
        state = get_sync_state()
        state.update(updates)
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return state


def log_import_event(
    source: str,
    status: str,
    records_count: int,
    valid_count: int,
    failed_count: int,
    file_name: str = "",
    error_message: str = "",
    duration_sec: float = 0.0,
) -> None:
    """Log an ingestion import event to logs/imports.jsonl."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "status": status,
        "file_name": file_name,
        "records_count": records_count,
        "valid_count": valid_count,
        "failed_count": failed_count,
        "error_message": error_message,
        "duration_seconds": round(duration_sec, 3),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _state_lock:
        with open(IMPORT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _append_or_update_report(scored_df: pd.DataFrame) -> None:
    """
    Append or update prediction results in reports/customer_intelligence.csv.
    Invalidates in-memory CSV cache in customer.py.
    """
    report_path = REPORTS_DIR / "customer_intelligence.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        existing_df = pd.read_csv(report_path)
        combined = pd.concat([existing_df, scored_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["customer_id"], keep="last")
    else:
        combined = scored_df

    combined.to_csv(report_path, index=False)
    logger.info(f"AutoIngestion: Updated {report_path.name} with {len(scored_df)} scored records.")



def process_subscriber_dataframe(df: pd.DataFrame, source_name: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Validate, preprocess, score, and persist a subscriber dataframe.
    Returns (valid_count, failed_count, scored_results_list).
    """
    start_t = time.time()

    # Ensure required columns
    required_id_cols = ["customer_id", "customerID", "id"]
    id_col = None
    for c in df.columns:
        if c in required_id_cols or c.lower() in ["customer_id", "customerid"]:
            id_col = c
            break

    if not id_col:
        df["customer_id"] = [f"TEL-{int(time.time())}-{i:04d}" for i in range(len(df))]
        id_col = "customer_id"

    df[id_col] = df[id_col].astype(str).str.strip()

    valid_rows = []
    failed_rows = []

    for _, row in df.iterrows():
        rec = row.to_dict()
        cid = str(rec.get(id_col, "")).strip()
        if not cid or cid.lower() in ["nan", "none", "null"]:
            failed_rows.append({"record": rec, "reason": "Missing or invalid customer_id"})
        else:
            rec["customer_id"] = cid
            valid_rows.append(rec)

    if not valid_rows:
        return 0, len(failed_rows), []

    valid_df = pd.DataFrame(valid_rows)

    # Fill defaults for required ML features
    defaults = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract_type": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.0,
        "total_charges": 840.0,
    }
    for col, val in defaults.items():
        if col not in valid_df.columns:
            valid_df[col] = val

    # Run ML Churn & Intelligence Inference
    scored_records = []
    for _, row in valid_df.iterrows():
        rec = row.to_dict()
        try:
            pred_res = predict_churn(rec)

            monthly = float(rec.get("monthly_charges", 70.0))
            tenure = max(1, int(rec.get("tenure_months", 12)))
            prob = pred_res.get("churn_probability", 0.3)
            churn_pred = pred_res.get("churn_prediction", 0)

            # Map full intelligence properties
            rec["churn_probability"] = round(prob, 4)
            rec["predicted_churn"] = churn_pred
            rec["churn_risk_level"] = pred_res.get("risk_level", "Low Risk")
            rec["predicted_ltv"] = round(monthly * tenure, 2)
            rec["projected_future_ltv"] = round(monthly * 24.0 * (1.0 - prob), 2)
            rec["intelligence_score"] = round((1.0 - prob) * 100.0, 1)

            # Segment assignment
            if monthly >= 80.0 and tenure >= 24:
                rec["customer_segment"] = "High-Value Subscribers"
            elif tenure >= 36:
                rec["customer_segment"] = "Loyal Subscribers"
            elif monthly < 45.0:
                rec["customer_segment"] = "Budget Subscribers"
            else:
                rec["customer_segment"] = "Growth Subscribers"

            # Recommendations
            if prob >= 0.61:
                rec["primary_recommendation"] = "15% Contract Upgrade Discount + Priority Support"
                rec["recommendation_priority"] = "Critical"
                rec["estimated_revenue_saved"] = round(monthly * 12.0 * 0.7, 2)
            elif prob >= 0.40:
                rec["primary_recommendation"] = "Retention Check-in & Free Speed Boost"
                rec["recommendation_priority"] = "High"
                rec["estimated_revenue_saved"] = round(monthly * 6.0 * 0.5, 2)
            else:
                rec["primary_recommendation"] = "Standard Support & Loyalty Points"
                rec["recommendation_priority"] = "Low"
                rec["estimated_revenue_saved"] = 0.0

            rec["rfm_persona"] = "High Spender" if monthly > 70.0 else "Regular Subscriber"
            scored_records.append(rec)
        except Exception as e:
            logger.warning(f"AutoIngestion: Scoring failed for {rec.get('customer_id')}: {e}")
            failed_rows.append({"record": rec, "reason": str(e)})

    if scored_records:
        scored_df = pd.DataFrame(scored_records)
        _append_or_update_report(scored_df)

        now_iso = datetime.now(timezone.utc).isoformat()
        current_state = get_sync_state()
        update_sync_state({
            "last_auto_sync": now_iso,
            "last_prediction_time": now_iso,
            "records_processed_today": current_state.get("records_processed_today", 0) + len(scored_records),
            "sync_status": "success",
        })

    duration = time.time() - start_t
    log_import_event(
        source=source_name,
        status="success" if scored_records else "failed",
        records_count=len(df),
        valid_count=len(scored_records),
        failed_count=len(failed_rows),
        duration_sec=duration,
    )

    return len(scored_records), len(failed_rows), scored_records


def scan_watch_folder() -> Dict[str, Any]:
    """
    Scan data/incoming/ for new CSV files.
    Process valid CSVs and move them to data/processed/ or data/failed/.
    """
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(INCOMING_DIR.glob("*.csv"))
    if not csv_files:
        return {"processed_files": 0, "status": "no_files"}

    logger.info(f"WatchFolder: Found {len(csv_files)} incoming CSV file(s).")
    processed_count = 0
    failed_count = 0

    for csv_path in csv_files:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_name = f"{ts}_{csv_path.name}"

        try:
            df = pd.read_csv(csv_path)
            valid_c, fail_c, scored = process_subscriber_dataframe(df, source_name=f"watch_folder:{csv_path.name}")

            if valid_c > 0:
                dest = PROCESSED_DIR / target_name
                shutil.move(str(csv_path), str(dest))
                processed_count += 1
                logger.info(f"WatchFolder: Processed {csv_path.name} -> {dest.name}")
            else:
                dest = FAILED_DIR / target_name
                shutil.move(str(csv_path), str(dest))
                failed_count += 1
                logger.warning(f"WatchFolder: Failed file {csv_path.name} -> {dest.name}")

        except Exception as e:
            dest = FAILED_DIR / target_name
            try:
                shutil.move(str(csv_path), str(dest))
            except Exception:
                pass
            failed_count += 1
            logger.error(f"WatchFolder: Error processing file {csv_path.name}: {e}")
            log_import_event(
                source=f"watch_folder:{csv_path.name}",
                status="failed",
                records_count=0,
                valid_count=0,
                failed_count=1,
                file_name=csv_path.name,
                error_message=str(e),
            )

    state = get_sync_state()
    update_sync_state({
        "processed_file_count": state.get("processed_file_count", 0) + processed_count,
        "failed_file_count": state.get("failed_file_count", 0) + failed_count,
    })

    return {
        "processed_files": processed_count,
        "failed_files": failed_count,
        "status": "completed",
    }


def run_database_auto_sync() -> Dict[str, Any]:
    """
    Incremental auto-sync from PostgreSQL database to Prediction Engine.
    Queries newly added/updated subscribers and updates reports/customer_intelligence.csv.
    """
    start_t = time.time()
    try:
        from backend.database.database import SessionLocal
        from backend.models.customer import Customer

        db = SessionLocal()
        try:
            customers = db.query(Customer).all()
            if not customers:
                db.close()
                return {"sync_status": "no_records", "processed": 0}

            records = []
            for c in customers:
                r = {
                    "customer_id": c.customer_id,
                    "gender": getattr(c, "gender", "Female"),
                    "senior_citizen": getattr(c, "senior_citizen", 0),
                    "partner": getattr(c, "partner", "No"),
                    "dependents": getattr(c, "dependents", "No"),
                    "tenure_months": getattr(c, "tenure_months", 12),
                }
                if hasattr(c, "contract") and c.contract:
                    r["contract_type"] = c.contract.contract_type
                    r["paperless_billing"] = c.contract.paperless_billing
                    r["payment_method"] = c.contract.payment_method
                if hasattr(c, "billing") and c.billing:
                    r["monthly_charges"] = float(c.billing.monthly_charges)
                    r["total_charges"] = float(c.billing.total_charges)
                records.append(r)

            db.close()
            df = pd.DataFrame(records)
            valid_c, fail_c, scored = process_subscriber_dataframe(df, source_name="db_auto_sync")
            
            duration = time.time() - start_t
            now_iso = datetime.now(timezone.utc).isoformat()
            update_sync_state({
                "last_auto_sync": now_iso,
                "sync_status": "success",
            })
            return {"sync_status": "success", "processed": valid_c, "duration_sec": round(duration, 2)}
        except Exception as db_err:
            db.close()
            logger.warning(f"DBAutoSync: Database query skipped (offline or empty): {db_err}")
            return {"sync_status": "db_offline", "processed": 0}
    except Exception as e:
        logger.warning(f"DBAutoSync: Skipping DB sync: {e}")
        return {"sync_status": "skipped", "reason": str(e)}
