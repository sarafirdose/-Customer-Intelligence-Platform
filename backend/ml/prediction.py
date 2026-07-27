"""
Model inference execution module.

Loads trained models from the artifacts registry to generate predictions on features.
"""

import os
from typing import Any
import joblib
import pandas as pd
from backend.core.logger import logger


class PredictionEngine:
    """
    Executes model loading and real-time inference.
    """

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        """
        Initialize the prediction engine.

        Args:
            artifacts_dir: Parent folder of artifacts.
        """
        self.models_dir = os.path.join(artifacts_dir, "models")
        self.model_path = os.path.join(self.models_dir, "churn_model.pkl")

    def _load_model(self) -> Any:
        """
        Load the classifier model from disk.

        Returns:
            Any: Trained model instance.
        """
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Model file {self.model_path} not found. Running with mock fallback."
            )
            return None
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model file: {e}")
            return None

    def predict_churn_probability(self, features: pd.DataFrame) -> float:
        """
        Generate churn probability index from engineered features.

        Args:
            features: Formatted DataFrame of features.

        Returns:
            float: Churn probability score.
        """
        model = self._load_model()
        if model is None:
            # Fallback mock calculation if model registry is not compiled yet
            return 0.28

        try:
            probs = model.predict_proba(features)
            # Return class 1 probability
            return float(probs[0, 1])
        except Exception as e:
            logger.error(f"Inference error occurred: {e}")
            return 0.28
