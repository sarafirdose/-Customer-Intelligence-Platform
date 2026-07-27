"""
Model Explainability (SHAP) module.

Generates Shapley feature attributions to provide explainable AI insights,
indicating which characteristics drive individual customer churn predictions.
"""

from typing import Any, Dict
import pandas as pd
from backend.core.logger import logger


class ModelExplainer:
    """
    Computes explainability insights for predictions using SHAP values.
    """

    def __init__(self, model: Any = None) -> None:
        """
        Initialize the explainer.

        Args:
            model: Trained classifier model instance.
        """
        self.model = model

    def explain_prediction(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Compute feature attribution weights for a given customer profile.

        Args:
            features: Single customer feature DataFrame.

        Returns:
            Dict[str, float]: Attribute name to impact weight dictionary.
        """
        logger.info("Computing SHAP feature attributions.")

        # In Phase 0, we return mock SHAP attributions reflecting general weights.
        # These will be updated to load a real shap.TreeExplainer in Phase 1.
        mock_shap_values = {
            "tenure_months": -0.18,
            "monthly_charges": 0.12,
            "total_charges": -0.04,
            "is_month_to_month": 0.22,
            "is_two_year": -0.15,
            "has_tech_support": -0.09,
        }

        # Filter to include only features present in input
        attributions = {
            col: mock_shap_values.get(col, 0.01)
            for col in features.columns
        }

        logger.info(f"Generated attributions for {len(attributions)} features.")
        return attributions
