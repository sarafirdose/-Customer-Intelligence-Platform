"""
Phase 8 Production Deployment, Scalable Inference & MLOps Endpoints.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, status
from fastapi.responses import JSONResponse

from backend.cache.prediction_cache import prediction_cache
from backend.core.alerts import alert_manager
from backend.ml.batch import batch_engine
from backend.ml.deployment_manager import deployment_manager
from backend.ml.explain import explainability_service
from backend.ml.prediction_store import prediction_store
from backend.ml.registry import promote
from backend.ml.retraining import retraining_pipeline
from backend.workers.queue_manager import queue_manager

router = APIRouter(tags=["deployment_and_scale"])


# ---------------------------------------------------------------------------
# Deployment & Canary / BlueGreen APIs
# ---------------------------------------------------------------------------

@router.get("/deployment/status", summary="Deployment Status")
def get_deployment_status() -> JSONResponse:
    """Return Blue/Green environment, Canary split, and production model state."""
    res = deployment_manager.get_deployment_status()
    return JSONResponse(content=res)


@router.post("/deployment/promote", summary="Promote Model")
def promote_model(
    model_name: str = Body(..., embed=True),
    version: str = Body(..., embed=True),
    target_status: str = Body("production", embed=True),
) -> JSONResponse:
    """Promote a model version to staging or production."""
    try:
        entry = promote(model_name, version, target_status)
        return JSONResponse(content={"status": "promoted", "entry": entry})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/deployment/rollback", summary="Rollback Model")
def rollback_model(model_name: str = Body("churn", embed=True)) -> JSONResponse:
    """Trigger instant rollback to previous production model version."""
    res = deployment_manager.execute_rollback(model_name)
    return JSONResponse(content=res)


@router.post("/deployment/canary", summary="Update Canary Split")
def update_canary(
    target_version: str = Body(..., embed=True),
    percentage: int = Body(..., embed=True, ge=0, le=100),
) -> JSONResponse:
    """Advance or update Canary traffic split percentage (0-100%)."""
    res = deployment_manager.update_canary(target_version, percentage)
    return JSONResponse(content=res)


@router.post("/deployment/bluegreen", summary="Switch Blue/Green Traffic")
def switch_bluegreen() -> JSONResponse:
    """Instantly switch active environment between Blue and Green."""
    res = deployment_manager.switch_bluegreen()
    return JSONResponse(content=res)


# ---------------------------------------------------------------------------
# Prediction Cache APIs
# ---------------------------------------------------------------------------

@router.get("/cache/stats", summary="Prediction Cache Metrics")
def get_cache_stats() -> JSONResponse:
    """Return hit count, miss count, hit ratio, and Redis status."""
    stats = prediction_cache.get_stats()
    return JSONResponse(content=stats)


# ---------------------------------------------------------------------------
# Batch & Async Prediction Queue APIs
# ---------------------------------------------------------------------------

@router.post("/predictions/batch", summary="Async Batch Prediction")
def run_batch_predictions(
    customers: List[Dict[str, Any]] = Body(...),
    background_tasks: BackgroundTasks = None,
) -> JSONResponse:
    """Enqueue bulk batch prediction job."""
    if not customers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty customer list")

    job_id = queue_manager.enqueue_batch(customers)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": f"Enqueued batch of {len(customers)} records.",
        },
    )


@router.get("/predictions/history", summary="Prediction Store History")
def get_prediction_history(
    customer_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    """Query historical predictions recorded in Prediction Store."""
    history = prediction_store.get_history(customer_id=customer_id, limit=limit)
    summary = prediction_store.get_analytics_summary()
    return JSONResponse(content={"history": history, "analytics_summary": summary})


# ---------------------------------------------------------------------------
# Explainability API
# ---------------------------------------------------------------------------

@router.get("/explain/{customer_id}", summary="SHAP Customer Explanation")
def get_customer_explanation(customer_id: str) -> JSONResponse:
    """Return SHAP local explanation, waterfall data, and key feature drivers."""
    explanation = explainability_service.explain_customer(customer_id=customer_id)
    global_importance = explainability_service.get_global_importance()
    return JSONResponse(content={"explanation": explanation, "global_importance": global_importance})


# ---------------------------------------------------------------------------
# Continuous Retraining APIs
# ---------------------------------------------------------------------------

@router.post("/retraining/run", summary="Trigger Continuous Retraining")
def trigger_retraining(
    dataset_path: Optional[str] = Body(None, embed=True),
    auto_promote: bool = Body(False, embed=True),
) -> JSONResponse:
    """Trigger automated continuous retraining workflow."""
    res = retraining_pipeline.run_retraining(
        dataset_path=dataset_path,
        auto_promote_production=auto_promote,
        trigger_type="manual_api",
    )
    return JSONResponse(content=res)


@router.get("/retraining/history", summary="Retraining History")
def get_retraining_history(limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
    """Return history of retraining pipeline runs."""
    history = retraining_pipeline.get_history(limit=limit)
    return JSONResponse(content={"history": history})


# ---------------------------------------------------------------------------
# Alerts API
# ---------------------------------------------------------------------------

@router.get("/alerts", summary="Alert History & Dispatcher")
def get_alerts(limit: int = Query(50, ge=1, le=200)) -> JSONResponse:
    """Return recent operational alerts."""
    history = alert_manager.get_alert_history(limit=limit)
    return JSONResponse(content={"alerts": history})
