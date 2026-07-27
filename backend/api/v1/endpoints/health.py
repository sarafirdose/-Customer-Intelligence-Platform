"""
Application Health Monitoring Endpoint.

Returns structural system health status, validating database connectivity
and service status for container orchestration (e.g. Kubernetes, Docker).
"""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.database.database import get_db, test_db_connection
from backend.core.settings import settings

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Detailed system health monitoring",
    description="Validates that both API server and databases are online and accessible.",
)
def check_health(db: Session = Depends(get_db)) -> JSONResponse:
    """
    Perform a live database query test and return full service health.

    Args:
        db: Scoped database session injection.

    Returns:
        JSONResponse: Detailed json document of the systems condition.
    """
    db_connected = test_db_connection()

    payload = {
        "status": "healthy" if db_connected else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "api": "running",
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
    }

    if not db_connected:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
