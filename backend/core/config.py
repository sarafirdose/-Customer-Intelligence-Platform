"""
Application core configurations and constants.

Defines static metadata, Swagger OpenAPI specifications, and constants used
across different modules of the Customer Intelligence Platform.
"""

from typing import Any, Dict

# OpenAPI swagger documentation configuration
API_TITLE: str = "Customer Churn Prediction & Lifetime Value (LTV) Engine"
API_DESCRIPTION: str = (
    "Enterprise-grade analytics system that predicts customer churn, estimates "
    "customer lifetime value (LTV), exposes REST APIs, and yields explainable AI "
    "insights using SHAP."
)
API_VERSION: str = "1.0.0"

# CORS configuration
CORS_ORIGINS: list[str] = [
    "http://localhost",
    "http://localhost:3000",  # Common frontend port
    "http://localhost:8000",  # Backend itself
    "http://localhost:8501",  # Streamlit dashboard default port
]

# ML Model Configuration defaults
MODEL_DEFAULTS: Dict[str, Any] = {
    "random_state": 42,
    "churn_threshold": 0.5,
    "test_size": 0.2,
}

# Superset analytics database connection configurations
SUPERSET_COMPATIBILITY_SETTINGS: Dict[str, Any] = {
    "expose_views": True,
    "view_schema_name": "analytics_views",
}
