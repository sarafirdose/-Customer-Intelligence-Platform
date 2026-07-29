"""
Application Health & Monitoring Endpoints.

GET /health  — Deep health check: DB + artifact presence
GET /ready   — Lightweight liveness probe (no DB query)
GET /metrics — Full metrics snapshot from MetricsCollector
"""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.core.audit import log_audit_event
from backend.core.metrics import metrics as metrics_collector
from backend.core.settings import settings
from backend.database.database import get_db, test_db_connection

router = APIRouter()

REQUIRED_ARTIFACTS = [
    "artifacts/models/best_model.pkl",
    "artifacts/models/preprocessor.pkl",
    "artifacts/models/metadata.json",
]


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Deep system health check",
    description="Validates DB connectivity and presence of required model artifacts.",
    tags=["monitoring"],
)
def check_health(db: Session = Depends(get_db)) -> JSONResponse:
    """
    Perform a live database query test and verify model artifacts exist.

    Returns HTTP 200 if healthy, 503 if degraded.
    """
    db_connected = test_db_connection()

    missing_artifacts = [
        f for f in REQUIRED_ARTIFACTS if not Path(f).exists()
    ]
    artifacts_ok = len(missing_artifacts) == 0

    overall_healthy = db_connected and artifacts_ok
    payload = {
        "status": "healthy" if overall_healthy else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "artifacts": "present" if artifacts_ok else f"missing: {missing_artifacts}",
        "api": "running",
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
    }

    log_audit_event(
        event_type="SYSTEM",
        endpoint="/health",
        result_summary={"status": payload["status"]},
    )

    if not overall_healthy:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Lightweight liveness probe",
    description="Returns 200 immediately — used by load balancers and container orchestrators.",
    tags=["monitoring"],
)
def check_ready() -> JSONResponse:
    """
    Fast liveness check — no I/O, no DB. Just confirms the process is alive.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "version": settings.APP_VERSION},
    )


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Application performance metrics",
    description=(
        "Returns current API request counts, latency percentiles, CPU/memory usage, "
        "prediction throughput, and batch job totals."
    ),
    tags=["monitoring"],
)
def get_metrics() -> JSONResponse:
    """
    Return a full metrics snapshot from the in-memory MetricsCollector.
    """
    if not settings.METRICS_ENABLED:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Metrics collection is disabled (METRICS_ENABLED=false)."},
        )
    snap = metrics_collector.snapshot()
    return JSONResponse(status_code=status.HTTP_200_OK, content=snap)
