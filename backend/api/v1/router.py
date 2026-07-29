"""
API Version 1 Routing Aggregation.

Combines endpoints for health validation, predictions, ingestion, customer intelligence,
observability, and production deployment/scale.
"""

from fastapi import APIRouter
from backend.api.v1.endpoints import (
    customer,
    deployment,
    health,
    ingest,
    observability,
    predict,
)

api_router = APIRouter()

# Register sub-routes
api_router.include_router(health.router, tags=["monitoring"])
api_router.include_router(predict.router, tags=["predictions"])
api_router.include_router(ingest.router, tags=["ingestion"])
api_router.include_router(customer.router, tags=["intelligence"])
api_router.include_router(observability.router, tags=["observability"])
api_router.include_router(deployment.router, prefix="/api/v1" if False else "", tags=["deployment_and_scale"])
