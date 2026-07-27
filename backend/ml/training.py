"""
Model training pipeline module.

Configures training loops for churn predictions and LTV estimations using classifiers
(e.g., XGBoost, LightGBM) and regression algorithms. Validates metrics and saves models.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from backend.core.logger import logger
from backend.core.settings import settings


class ModelTrainer:
    """
    Orchestrates the model training, testing, and serialization pipeline.
    """

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        """
        Initialize the trainer with targets registry paths.

        Args:
            artifacts_dir: Folder containing model artifacts.
        """
        self.artifacts_dir = artifacts_dir
        self.models_dir = os.path.join(artifacts_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def train_churn_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a Churn classification model (RandomForest/XGBoost placeholder).

        Args:
            X: Matrix of features.
            y: Binary target series (churn labels).

        Returns:
            Tuple[Any, Dict[str, float]]: Trained model object, dictionary of metrics.
        """
        logger.info("Initializing Churn Model Training.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # Using RandomForestClassifier as standard starting algorithm
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        logger.info("Fitting classification model to training split.")
        model.fit(X_train, y_train)

        # Predictions for validation
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "auc": float(roc_auc_score(y_test, probs)) if len(y_test.unique()) > 1 else 1.0,
        }

        logger.info(f"Training completed successfully. Validation Metrics: {metrics}")

        # Serialize model artifact
        model_save_path = os.path.join(self.models_dir, "churn_model.pkl")
        joblib.dump(model, model_save_path)
        logger.info(f"Model saved to registry location: {model_save_path}")

        return model, metrics
