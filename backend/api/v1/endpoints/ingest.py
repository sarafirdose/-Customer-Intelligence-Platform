"""
REST API Endpoints for Data Ingestion.

Exposes endpoints to trigger dataset ingestion and check ingestion history
and database population metrics.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.services.ingestion_service import IngestionService
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
    """
    Run the data ingestion pipeline synchronously and return metrics.

    Args:
        db: Scoped database session.

    Returns:
        Dict[str, Any]: Summary metrics of the ingestion run.
    """
    service = IngestionService(db)
    result = service.run_pipeline()

    if result.get("status") == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result
        )

    return result


@router.get(
    "/dataset/status",
    status_code=status.HTTP_200_OK,
    summary="Retrieve ingestion status and database metrics",
)
def get_dataset_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Check dataset CSV file presence and query record counters.

    Args:
        db: Scoped database session.

    Returns:
        Dict[str, Any]: Mapping showing CSV availability, DB counts, and latest import timestamps.
    """
    extractor = DataExtractor()
    csv_available = extractor.is_dataset_available()

    try:
        record_count = db.query(Customer).count()
        db_populated = record_count > 0
    except Exception:
        # Fallback if database table is not created yet
        record_count = 0
        db_populated = False

    # Resolve latest import time
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
    }
