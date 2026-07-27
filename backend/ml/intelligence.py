"""
Customer Intelligence Platform - Analytics Engine.

Implements:
1. LTV Regression (Linear Regression, Random Forest, XGBoost, LightGBM)
2. K-Means Customer Clustering (Silhouette/Elbow Optimal K Selection)
3. RFM Analysis (Recency, Frequency, Monetary calculations)
4. Configurable Customer Intelligence Score (0-100)
5. Hybrid Recommendation Engine (ML + Rules + SHAP logic)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb

from backend.core.logger import logger
from backend.core.settings import settings
from backend.ml.training import engineer_features

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_REGISTRY = ARTIFACT_DIR / "models"


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_ltv_models(df: pd.DataFrame) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Train and compare multiple LTV regressors on Total Charges.
    """
    logger.info("LTV Pipeline: Starting regression model training.")

    X = df.drop(columns=["id", "customer_id", "total_charges", "total_charges_log", "churn"], errors="ignore")
    y = df["total_charges"]

    # Train/test split 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Set up preprocessing
    numeric_features = ["tenure_months", "monthly_charges", "charges_ratio", "total_services"]
    categorical_features = [
        "contract_type", "payment_method", "internet_service", "tenure_group",
        "multiple_lines", "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies"
    ]
    binary_features = ["gender", "partner", "dependents", "phone_service", "paperless_billing"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), binary_features),
        ],
        remainder="drop"
    )

    # Fit preprocessor
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Regressors
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
        "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1),
    }

    comparison_records = []
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_trans, y_train)
        duration = time.time() - t0

        y_pred = model.predict(X_test_trans)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        comparison_records.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "Training Time (s)": duration
        })
        trained_models[name] = model

    comp_df = pd.DataFrame(comparison_records)
    comp_df = comp_df.sort_values(by="R2", ascending=False)

    best_model_name = comp_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    logger.info(f"LTV Pipeline: Best regressor model chosen: {best_model_name}")

    # Build LTV pipeline (preprocessor + model)
    ltv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", best_model)
    ])

    return ltv_pipeline, {
        "best_model": best_model_name,
        "rmse": float(comp_df.iloc[0]["RMSE"]),
        "mae": float(comp_df.iloc[0]["MAE"]),
        "mape": float(comp_df.iloc[0]["MAPE"]),
        "r2": float(comp_df.iloc[0]["R2"]),
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }, comp_df


def run_customer_segmentation(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Run K-Means Customer Clustering.
    """
    logger.info("Segmentation Pipeline: Running clustering calculations.")

    numeric_cols = ["tenure_months", "monthly_charges", "total_services", "charges_ratio"]
    X = df[numeric_cols].copy()

    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal K (2 to 6)
    k_range = range(2, 7)
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Optimal K is the one maximizing silhouette score
    optimal_k = k_range[np.argmax(silhouettes)]
    # Default to 4 clusters to correspond to Platinum, Gold, Silver, Bronze if silhouette is inconclusive
    if optimal_k < 3:
        optimal_k = 4

    logger.info(f"Segmentation Pipeline: Optimal cluster K selected: {optimal_k}")

    # Final KMeans fit
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_labels = kmeans.fit_predict(X_scaled)

    segmentation_pipeline = Pipeline([
        ("scaler", scaler),
        ("kmeans", kmeans)
    ])

    # Assign business labels based on median monetary spend (total_charges)
    df_temp = df.copy()
    df_temp["cluster_raw"] = df_labels

    cluster_charges = df_temp.groupby("cluster_raw")["total_charges"].median().sort_values(ascending=False)
    # Platinum, Gold, Silver, Bronze ranking
    segment_names = ["Platinum", "Gold", "Silver", "Bronze"]
    cluster_mapping = {}
    for idx, (cluster_id, _) in enumerate(cluster_charges.items()):
        # Fallback names if optimal_k > 4
        name = segment_names[idx] if idx < len(segment_names) else f"Bronze Class {idx - 2}"
        cluster_mapping[cluster_id] = name

    # Silhouette statistics dataframe
    stats_k_df = pd.DataFrame({
        "k": list(k_range),
        "inertia": inertias,
        "silhouette_score": silhouettes
    })

    return segmentation_pipeline, stats_k_df, pd.Series(df_labels).map(cluster_mapping)


def calculate_rfm(df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
    """
    Perform RFM Analysis.
    Recency: binned from (1.0 - Churn Probability) * 5
    Frequency: binned from tenure_months
    Monetary: binned from total_charges
    """
    df_rfm = pd.DataFrame(index=df.index)
    df_rfm["customer_id"] = df["customer_id"]

    # 1. Recency Score (1-5): High stay probability = High recency score
    stay_prob = 1.0 - churn_probs
    # Bin stay probabilities into quintiles (cut safely)
    try:
        df_rfm["R_score"] = pd.qcut(stay_prob, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except Exception:
        df_rfm["R_score"] = pd.cut(stay_prob, bins=5, labels=[1, 2, 3, 4, 5])

    # 2. Frequency Score (1-5): Binned from tenure_months
    try:
        df_rfm["F_score"] = pd.qcut(df["tenure_months"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except Exception:
        df_rfm["F_score"] = pd.cut(df["tenure_months"], bins=5, labels=[1, 2, 3, 4, 5])

    # 3. Monetary Score (1-5): Binned from total_charges
    try:
        df_rfm["M_score"] = pd.qcut(df["total_charges"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except Exception:
        df_rfm["M_score"] = pd.cut(df["total_charges"], bins=5, labels=[1, 2, 3, 4, 5])

    # Cast to integer
    df_rfm["R_score"] = df_rfm["R_score"].astype(int)
    df_rfm["F_score"] = df_rfm["F_score"].astype(int)
    df_rfm["M_score"] = df_rfm["M_score"].astype(int)

    # Combined RFM Score
    df_rfm["rfm_score"] = df_rfm["R_score"] * 100 + df_rfm["F_score"] * 10 + df_rfm["M_score"]

    # Assign Personas
    def assign_persona(row) -> str:
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 4:
            return "Loyal Customers"
        elif r >= 3 and f >= 3 and m < 4:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Lost Customers"
        else:
            return "About to Sleep"

    df_rfm["persona"] = df_rfm.apply(assign_persona, axis=1)
    return df_rfm


def calculate_intelligence_score(
    churn_prob: float,
    predicted_ltv: float,
    tenure: int,
    services_count: int,
    max_ltv: float = 8500.0
) -> Tuple[float, str]:
    """
    Calculate composite Customer Intelligence Score (0-100) using configurable weights.
    Categories: Excellent (>=80), Good (>=60), Moderate (>=40), Poor (>=20), Critical (<20)
    """
    weights_path = ARTIFACT_DIR / "intelligence_weights.json"
    if weights_path.exists():
        with open(weights_path, "r") as f:
            weights = json.load(f)
    else:
        weights = {
            "churn_weight": 0.30,
            "ltv_weight": 0.30,
            "tenure_weight": 0.20,
            "services_weight": 0.20
        }

    # Normalize metrics to 0-100
    # Churn metric: higher score means LOWER churn probability
    c_score = (1.0 - churn_prob) * 100

    # LTV metric: min-max scaled log representation
    log_ltv = np.log1p(predicted_ltv)
    log_max_ltv = np.log1p(max_ltv)
    l_score = min(100.0, (log_ltv / log_max_ltv) * 100)

    # Tenure: max 72 months binned
    t_score = min(100.0, (tenure / 72.0) * 100)

    # Services: max 8 services binned
    s_score = min(100.0, (services_count / 8.0) * 100)

    score = (
        weights["churn_weight"] * c_score +
        weights["ltv_weight"] * l_score +
        weights["tenure_weight"] * t_score +
        weights["services_weight"] * s_score
    )
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 80:
        cat = "Excellent"
    elif score >= 60:
        cat = "Good"
    elif score >= 40:
        cat = "Moderate"
    elif score >= 20:
        cat = "Poor"
    else:
        cat = "Critical"

    return score, cat


def generate_recommendation_details(
    sample: Dict[str, Any],
    churn_prob: float,
    predicted_ltv: float,
    segment: str,
    persona: str,
    shap_top_contrib: str = ""
) -> List[Dict[str, Any]]:
    """
    Rule-based Hybrid Recommendation Engine with SHAP annotations and estimated revenue saved.
    """
    recs = []

    # Helper traits
    contract = str(sample.get("contract_type", sample.get("Contract", ""))).strip()
    tech_support = str(sample.get("tech_support", sample.get("TechSupport", ""))).strip()
    internet = str(sample.get("internet_service", sample.get("InternetService", ""))).strip()
    is_auto = 1 if "automatic" in str(sample.get("payment_method", sample.get("PaymentMethod", ""))).lower() else 0
    tenure = int(sample.get("tenure_months", sample.get("tenure", 0)))
    total_services = int(sample.get("total_services", 0))

    # 1. Rule 1: High Churn Risk + High Value -> Premium Retention Package
    if churn_prob >= 0.40 and predicted_ltv >= 3500.0:
        revenue_saved = float(churn_prob * predicted_ltv)
        recs.append({
            "recommendation": "Offer Premium Retention Package",
            "priority": "Critical",
            "confidence": float(churn_prob),
            "reason": [
                "Customer has High Churn Risk (" + f"{churn_prob*100:.1f}%" + ")",
                "High customer lifetime valuation (" + f"${predicted_ltv:.2f}" + ")",
                f"SHAP indicator highlights key churn driver: {shap_top_contrib}" if shap_top_contrib else "High value retention target"
            ],
            "estimated_revenue_saved": revenue_saved
        })

    # 2. Rule 2: Month-to-Month Contract -> Recommend Annual Contract Upgrade
    if "month-to-month" in contract.lower():
        # F1-score threshold improvement
        revenue_saved = float(churn_prob * predicted_ltv * 0.45)
        recs.append({
            "recommendation": "Recommend Annual Contract Upgrade",
            "priority": "High",
            "confidence": 0.85,
            "reason": [
                "Flexible Month-to-month contract structure is active",
                "Locking into 1-year contract reduces churn probability by over 70%"
            ],
            "estimated_revenue_saved": revenue_saved
        })

    # 3. Rule 3: Low Support Add-ons -> Tech Support Promotion
    if "no" in tech_support.lower() and "no" not in internet.lower():
        revenue_saved = float(churn_prob * predicted_ltv * 0.15)
        recs.append({
            "recommendation": "Offer Proactive Tech Support Promotion",
            "priority": "Medium",
            "confidence": 0.70,
            "reason": [
                "High-speed internet is active but customer lacks Tech Support services",
                "Support subscribers show a 60% reduction in attrition rates"
            ],
            "estimated_revenue_saved": revenue_saved
        })

    # 4. Rule 4: Low service bundling -> Service Bundling Promotion
    if total_services <= 2 and "no" not in internet.lower():
        revenue_saved = float(churn_prob * predicted_ltv * 0.10)
        recs.append({
            "recommendation": "Recommend Service Bundling Upgrade",
            "priority": "Low",
            "confidence": 0.60,
            "reason": [
                f"Active account has low service density ({total_services} services)",
                "Bundling online backup or security services creates account lock-in"
            ],
            "estimated_revenue_saved": revenue_saved
        })

    # 5. Rule 5: Manual payment methods -> Automatic Billing Setup
    if is_auto == 0:
        revenue_saved = float(churn_prob * predicted_ltv * 0.20)
        recs.append({
            "recommendation": "Automatic Billing Setup Promotion",
            "priority": "Medium",
            "confidence": 0.75,
            "reason": [
                "Manual billing payment method is active",
                "Auto-pay customers exhibit a 66% lower attrition rate than manual check payers"
            ],
            "estimated_revenue_saved": revenue_saved
        })

    # Default if no rules triggered
    if not recs:
        recs.append({
            "recommendation": "Standard Customer Loyalty Outreach",
            "priority": "Low",
            "confidence": 0.50,
            "reason": ["Account is highly stable and active"],
            "estimated_revenue_saved": 0.0
        })

    # Sort recommendations by priority (Critical, High, Medium, Low)
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recs = sorted(recs, key=lambda x: priority_order.get(x["priority"], 4))

    return recs
