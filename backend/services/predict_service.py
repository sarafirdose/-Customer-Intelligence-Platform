"""
ML Inference Prediction Service.

Exposes interfaces to load trained models/preprocessors, run preprocessing
transformations, perform prediction inferences, compute latencies, and classify
risk confidence levels.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from backend.core.logger import logger
from backend.ml.training import engineer_features

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = BASE_DIR / "models"


def load_model() -> Any:
    """
    Load the best trained model pickle.
    """
    model_path = MODEL_REGISTRY / "best_model.pkl"
    if not model_path.exists():
        # Fallback to artifacts registry
        model_path = BASE_DIR / "artifacts" / "models" / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    return joblib.load(model_path)


def load_preprocessor() -> Any:
    """
    Load the fitted scikit-learn ColumnTransformer preprocessor.
    """
    prep_path = MODEL_REGISTRY / "preprocessor.pkl"
    if not prep_path.exists():
        # Fallback to artifacts registry
        prep_path = BASE_DIR / "artifacts" / "models" / "preprocessor.pkl"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {prep_path}")
    return joblib.load(prep_path)


def load_metadata() -> Dict[str, Any]:
    """
    Load model performance and threshold metadata.
    """
    meta_path = MODEL_REGISTRY / "model_metadata.json"
    if not meta_path.exists():
        # Fallback to artifacts registry
        meta_path = BASE_DIR / "artifacts" / "models" / "metadata.json"
    if not meta_path.exists():
        # Default fallback
        return {"optimal_threshold": 0.5}
    with open(meta_path, "r") as f:
        return json.load(f)


def map_input_schema(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps Kaggle dataset casing (PascalCase/CamelCase) to standardized database casing.
    """
    mapping = {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure_months",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract_type",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    }
    mapped = {}
    for k, v in sample.items():
        db_key = mapping.get(k, k)
        mapped[db_key] = v
    return mapped


class PredictService:
    """
    Handles end-to-end model inference scoring.
    """

    def __init__(self):
        self.model = load_model()
        self.preprocessor = load_preprocessor()
        self.metadata = load_metadata()
        self.threshold = self.metadata.get("optimal_threshold", 0.5)
        # Extract features expected by the preprocessing pipeline
        self.feature_columns = self.metadata.get("features", [
            "gender", "senior_citizen", "partner", "dependents", "tenure_months",
            "phone_service", "multiple_lines", "internet_service", "online_security",
            "online_backup", "device_protection", "tech_support", "streaming_tv",
            "streaming_movies", "contract_type", "paperless_billing", "payment_method",
            "monthly_charges", "total_charges"
        ])

    def predict_proba(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability of churn and measure latency.
        """
        t_start = time.perf_counter()

        # 1. Standardise and cast schema
        mapped_sample = map_input_schema(sample)
        df_raw = pd.DataFrame([mapped_sample])

        # Cast charges to float
        if "total_charges" in df_raw.columns:
            df_raw["total_charges"] = pd.to_numeric(df_raw["total_charges"], errors="coerce").fillna(0.0)
        if "monthly_charges" in df_raw.columns:
            df_raw["monthly_charges"] = pd.to_numeric(df_raw["monthly_charges"], errors="coerce").fillna(0.0)
        if "tenure_months" in df_raw.columns:
            df_raw["tenure_months"] = pd.to_numeric(df_raw["tenure_months"], errors="coerce").fillna(0).astype(int)

        # 2. Engineer features
        df_eng = engineer_features(df_raw)

        # 3. Align features
        # Keep only the features expected by the training transformer
        df_final = df_eng[self.feature_columns]
        t_preprocess = time.perf_counter()

        # 4. Transform features
        X_trans = self.preprocessor.transform(df_final)

        # 5. Predict probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        t_end = time.perf_counter()

        prep_latency = (t_preprocess - t_start) * 1000.0
        pred_latency = (t_end - t_preprocess) * 1000.0
        total_latency = (t_end - t_start) * 1000.0

        return {
            "probability": prob,
            "latency": {
                "preprocessing_ms": float(prep_latency),
                "prediction_ms": float(pred_latency),
                "total_ms": float(total_latency),
            },
        }

    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict binary classification label, probability, confidence level, and latency.
        """
        proba_res = self.predict_proba(sample)
        prob = proba_res["probability"]

        # Classification based on optimal threshold
        prediction = 1 if prob >= self.threshold else 0

        # Categorize confidence level
        if prob >= 0.90:
            confidence = "Very High"
            risk_label = "Very High Risk"
        elif prob >= 0.75:
            confidence = "High"
            risk_label = "High Risk"
        elif prob >= 0.60:
            confidence = "Medium"
            risk_label = "Medium Risk"
        elif prob >= self.threshold:
            confidence = "Low"
            risk_label = "Low Risk"
        else:
            confidence = "High" if prob < 0.15 else "Medium"
            risk_label = "Minimal Risk"

        return {
            "prediction": prediction,
            "probability": prob,
            "threshold": self.threshold,
            "risk_level": risk_label,
            "confidence": confidence,
            "latency": proba_res["latency"],
        }


# Global singleton instance for helper functions
_predict_service_instance = None


def get_predict_service() -> PredictService:
    global _predict_service_instance
    if _predict_service_instance is None:
        _predict_service_instance = PredictService()
    return _predict_service_instance


def predict_churn(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level helper to score customer churn."""
    from backend.ml.feature_store import feature_store
    full_sample = feature_store.apply_defaults(sample)

    service = get_predict_service()
    res = service.predict(full_sample)
    # Ensure standard keys for downstream consumers
    prob = res.get("probability", 0.0)
    pred = res.get("prediction", 0)
    risk = res.get("risk_level", "Low")
    ltv = float(full_sample.get("monthly_charges", 65.0)) * max(1, float(full_sample.get("tenure_months", 12)))
    return {
        "churn_probability": prob,
        "churn_prediction": pred,
        "risk_level": risk,
        "segment": "Standard",
        "predicted_ltv": ltv,
        "intelligence_score": round((1.0 - prob) * 100, 1),
        "recommendations": {"action": "Proactive Retention" if pred == 1 else "Standard Support"},
    }

