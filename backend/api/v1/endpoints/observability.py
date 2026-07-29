"""
System Observability Endpoints.

GET /registry               — Full model registry listing
GET /registry/{model}       — All versions of a specific model
GET /registry/{model}/production — Active production model
GET /registry/{model}/compare   — Side-by-side metric comparison (?v1=...&v2=...)
GET /config                 — Non-sensitive settings dump
GET /sysinfo                — OS, Python, disk, process info
GET /scheduler/jobs         — Scheduled job status
"""

import platform
import sys
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from backend.core.audit import log_audit_event
from backend.core.scheduler import platform_scheduler
from backend.core.settings import settings
from backend.ml.registry import (
    compare_versions,
    get_production_model,
    list_all_models,
    list_versions,
)

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get(
    "/registry",
    summary="Full model registry",
    description="Returns all registered model versions across all model types.",
)
def get_full_registry() -> JSONResponse:
    """Return the complete model registry."""
    registry = list_all_models()
    log_audit_event(event_type="SYSTEM", endpoint="/observability/registry")
    return JSONResponse(content=registry)


@router.get(
    "/registry/{model_name}",
    summary="All versions of a model",
)
def get_model_versions(model_name: str) -> JSONResponse:
    """Return all versions of a specific model, newest first."""
    versions = list_versions(model_name)
    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No versions found for model '{model_name}'.",
        )
    return JSONResponse(content={"model_name": model_name, "versions": versions})


@router.get(
    "/registry/{model_name}/production",
    summary="Active production model",
)
def get_production(model_name: str) -> JSONResponse:
    """Return the current production model entry."""
    prod = get_production_model(model_name)
    if prod is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No production model registered for '{model_name}'.",
        )
    return JSONResponse(content=prod)


@router.get(
    "/registry/{model_name}/compare",
    summary="Compare two model versions",
    description="Returns side-by-side metric diff. Pass ?v1=v1.0.0&v2=v1.1.0",
)
def compare_model_versions(
    model_name: str,
    v1: str = Query(..., description="First version string, e.g. v1.0.0"),
    v2: str = Query(..., description="Second version string, e.g. v1.1.0"),
) -> JSONResponse:
    """Side-by-side metric comparison between two registered versions."""
    try:
        result = compare_versions(model_name, v1, v2)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    log_audit_event(
        event_type="SYSTEM",
        endpoint=f"/observability/registry/{model_name}/compare",
        result_summary={"v1": v1, "v2": v2},
    )
    return JSONResponse(content=result)


@router.get(
    "/config",
    summary="Non-sensitive configuration dump",
    description="Returns current application settings, excluding secrets.",
)
def get_config() -> JSONResponse:
    """Dump safe (non-secret) configuration values."""
    safe_config = {
        "env": settings.ENV,
        "app_version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "log_level": settings.LOG_LEVEL,
        "log_json": settings.LOG_JSON,
        "metrics_enabled": settings.METRICS_ENABLED,
        "scheduler_enabled": settings.SCHEDULER_ENABLED,
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "max_request_size_mb": settings.MAX_REQUEST_SIZE_MB,
        "drift_warning_threshold": settings.DRIFT_WARNING_THRESHOLD,
        "drift_critical_threshold": settings.DRIFT_CRITICAL_THRESHOLD,
        "api_host": settings.API_HOST,
        "api_port": settings.API_PORT,
        # Sensitive fields intentionally omitted:
        # SECRET_KEY, DB_PASSWORD, DB_USER are never exposed
    }
    return JSONResponse(content=safe_config)


@router.get(
    "/sysinfo",
    summary="System information",
    description="Returns OS, Python version, process stats, and disk usage.",
)
def get_sysinfo() -> JSONResponse:
    """Return system-level diagnostics."""
    proc = psutil.Process()
    disk = psutil.disk_usage(".")

    sysinfo = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": {
            "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 1),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 1),
            "process_rss_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
        },
        "disk": {
            "total_gb": round(disk.total / 1024**3, 2),
            "used_gb": round(disk.used / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
            "percent_used": disk.percent,
        },
        "process_pid": proc.pid,
    }
    return JSONResponse(content=sysinfo)


@router.get(
    "/scheduler/jobs",
    summary="Scheduler job status",
    description="Returns all registered background jobs and their next scheduled run times.",
)
def get_scheduler_jobs() -> JSONResponse:
    """Return registered APScheduler jobs."""
    jobs = platform_scheduler.get_job_status()
    return JSONResponse(content={
        "scheduler_running": platform_scheduler.is_running,
        "jobs": jobs,
    })
