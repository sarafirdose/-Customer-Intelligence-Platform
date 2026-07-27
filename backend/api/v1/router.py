"""
API Version 1 Routing Aggregation.

Combines endpoints for health validation, predictions, and model explanations
under a structured v1 version prefix.
"""

from fastapi import APIRouter
from backend.api.v1.endpoints import health, predict, ingest, customer

api_router = APIRouter()

# Register sub-routes
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, tags=["predictions"])
api_router.include_router(ingest.router, tags=["ingestion"])
api_router.include_router(customer.router, tags=["intelligence"])
