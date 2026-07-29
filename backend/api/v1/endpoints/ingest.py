"""
REST API Endpoints for Real-Time & Automated Data Ingestion.

Exposes endpoints for:
  - Single record real-time ingestion (POST /ingest/record)
  - Batch records ingestion (POST /ingest/batch)
  - Full ETL pipeline execution (POST /ingest)
  - Dataset status & live auto-sync telemetry (GET /dataset/status, GET /ingest/state)
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
import pandas as pd
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.ingestion_service import IngestionService
from backend.services.auto_ingestion import (
    process_subscriber_dataframe,
    get_sync_state,
    scan_watch_folder,
)
from backend.models.customer import Customer
from backend.models.import_history import ImportHistory
from backend.ml.extractor import DataExtractor

router = APIRouter()


@router.post(
    "/ingest",
    status_code=status.HTTP_200_OK,
    summary="Trigger the customer dataset ETL ingestion pipeline",
    description="Loads raw data, profiles it, runs Pandera validation, cleans anomalies, and loads records into PostgreSQL.",
)
def run_ingestion(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Run the data ingestion pipeline synchronously and return metrics."""
    service = IngestionService(db)
    result = service.run_pipeline()

    if result.get("status") == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result
        )

    return result


@router.post(
    "/ingest/record",
    status_code=status.HTTP_200_OK,
    summary="Real-time ingestion for a single subscriber record",
    description="Validates input, stores record, runs churn/LTV/segment prediction, updates database & report CSV, and returns live predictions.",
)
def ingest_single_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest, validate, score, and store a single subscriber record."""
    if not record:
        raise HTTPException(status_code=400, detail="Empty record payload provided.")

    df = pd.DataFrame([record])
    valid_c, fail_c, scored = process_subscriber_dataframe(df, source_name="api:record")

    if not scored:
        raise HTTPException(
            status_code=422,
            detail="Failed to validate or score provided subscriber record."
        )

    return {
        "status": "success",
        "message": "Subscriber record successfully ingested and predictions generated.",
        "prediction": scored[0],
    }


@router.post(
    "/ingest/batch",
    status_code=status.HTTP_200_OK,
    summary="Real-time batch ingestion for multiple subscriber records",
    description="Validates a list of subscriber JSON objects, runs predictions, and updates reporting stores.",
)
def ingest_batch_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ingest, validate, score, and store a list of subscriber records in bulk."""
    if not records:
        raise HTTPException(status_code=400, detail="Empty records batch provided.")

    df = pd.DataFrame(records)
    valid_c, fail_c, scored = process_subscriber_dataframe(df, source_name="api:batch")

    return {
        "status": "success",
        "total_records": len(records),
        "valid_processed": valid_c,
        "failed_records": fail_c,
        "scored_predictions": scored[:5],  # Preview first 5
    }


@router.get(
    "/ingest/state",
    status_code=status.HTTP_200_OK,
    summary="Get live telemetry status of automated sync & watch folder pipeline",
)
def get_auto_ingest_state() -> Dict[str, Any]:
    """Retrieve auto-sync timestamps, processed counts, and watch folder state."""
    return get_sync_state()


@router.post(
    "/ingest/watch_folder/scan",
    status_code=status.HTTP_200_OK,
    summary="Manually trigger a watch folder scan of data/incoming/",
)
def trigger_watch_folder_scan() -> Dict[str, Any]:
    """Trigger an immediate scan of incoming CSV files."""
    return scan_watch_folder()


@router.get(
    "/dataset/status",
    status_code=status.HTTP_200_OK,
    summary="Retrieve ingestion status and database metrics",
)
def get_dataset_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Check dataset CSV file presence and query record counters."""
    extractor = DataExtractor()
    csv_available = extractor.is_dataset_available()

    try:
        record_count = db.query(Customer).count()
        db_populated = record_count > 0
    except Exception:
        record_count = 0
        db_populated = False

    try:
        latest_import = (
            db.query(ImportHistory)
            .filter(ImportHistory.status == "success")
            .order_by(ImportHistory.completed_at.desc())
            .first()
        )
        last_import_time = (
            latest_import.completed_at.isoformat() if latest_import else None
        )
    except Exception:
        last_import_time = None

    return {
        "dataset_available": csv_available,
        "database_populated": db_populated,
        "record_count": record_count,
        "last_import_time": last_import_time,
        "auto_sync_state": get_sync_state(),
    }
