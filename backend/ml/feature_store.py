"""
Feature Store & Feature Governance Engine.

Provides schema validation, feature versioning (e.g. 'v1.0'), feature lookup,
and training-serving feature consistency validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.core.logger import logger

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_STORE_JSON = BASE_DIR / "artifacts" / "feature_store.json"

# Feature Definitions
FEATURE_SCHEMA_V1: Dict[str, Dict[str, Any]] = {
    "gender": {"type": "categorical", "allowed": ["Male", "Female"], "default": "Male"},
    "senior_citizen": {"type": "numeric", "min": 0, "max": 1, "default": 0},
    "partner": {"type": "categorical", "allowed": ["Yes", "No"], "default": "No"},
    "dependents": {"type": "categorical", "allowed": ["Yes", "No"], "default": "No"},
    "tenure_months": {"type": "numeric", "min": 0, "max": 120, "default": 1},
    "contract_type": {"type": "categorical", "allowed": ["Month-to-month", "One year", "Two year"], "default": "Month-to-month"},
    "paperless_billing": {"type": "categorical", "allowed": ["Yes", "No"], "default": "Yes"},
    "payment_method": {
        "type": "categorical",
        "allowed": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "default": "Electronic check"
    },
    "phone_service": {"type": "categorical", "allowed": ["Yes", "No"], "default": "Yes"},
    "multiple_lines": {"type": "categorical", "allowed": ["Yes", "No", "No phone service"], "default": "No"},
    "internet_service": {"type": "categorical", "allowed": ["DSL", "Fiber optic", "No"], "default": "Fiber optic"},
    "online_security": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "online_backup": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "device_protection": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "tech_support": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "streaming_tv": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "streaming_movies": {"type": "categorical", "allowed": ["Yes", "No", "No internet service"], "default": "No"},
    "monthly_charges": {"type": "numeric", "min": 0.0, "max": 500.0, "default": 65.0},
    "total_charges": {"type": "numeric", "min": 0.0, "max": 50000.0, "default": 65.0},
}


class FeatureStore:
    """Feature Store Manager providing feature governance and schema validation."""

    def __init__(self, version: str = "v1.0"):
        self.version = version
        self.schema = FEATURE_SCHEMA_V1
        self.feature_names = list(self.schema.keys())

    def validate_features(self, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate incoming customer payload against feature store schema.

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []

        for feature_name, spec in self.schema.items():
            if feature_name not in payload or payload[feature_name] is None:
                errors.append(f"Missing required feature: '{feature_name}'")
                continue

            val = payload[feature_name]

            if spec["type"] == "numeric":
                try:
                    num_val = float(val)
                    if "min" in spec and num_val < spec["min"]:
                        errors.append(f"Feature '{feature_name}' value {num_val} < min {spec['min']}")
                    if "max" in spec and num_val > spec["max"]:
                        errors.append(f"Feature '{feature_name}' value {num_val} > max {spec['max']}")
                except (ValueError, TypeError):
                    errors.append(f"Feature '{feature_name}' expected numeric, got '{val}'")

            elif spec["type"] == "categorical":
                val_str = str(val)
                if "allowed" in spec and val_str not in spec["allowed"]:
                    errors.append(
                        f"Feature '{feature_name}' value '{val_str}' not in allowed values: {spec['allowed']}"
                    )

        return (len(errors) == 0, errors)

    def apply_defaults(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing features in payload with feature store defaults."""
        cleaned = dict(payload)
        for feature_name, spec in self.schema.items():
            if feature_name not in cleaned or cleaned[feature_name] is None:
                cleaned[feature_name] = spec["default"]
        return cleaned

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return Feature Store metadata description."""
        return {
            "version": "v1.0",
            "feature_count": len(FEATURE_SCHEMA_V1),
            "features": list(FEATURE_SCHEMA_V1.keys()),
            "schema_details": FEATURE_SCHEMA_V1,
        }

    def get_feature_vector(self, payload: Dict[str, Any]) -> List[Any]:
        """Return an ordered feature vector for model input."""
        cleaned = self.apply_defaults(payload)
        return [cleaned[feat] for feat in self.feature_names]


# Global FeatureStore instance
feature_store = FeatureStore()
