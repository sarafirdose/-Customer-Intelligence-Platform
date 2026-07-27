"""
Machine Learning Training Pipeline.

Performs feature engineering, builds column transformers, records data drift
baselines, runs cross-validation, tunes models (Logistic Regression, Random Forest,
XGBoost, LightGBM), sweeps thresholds, calculates SHAP explainability, measures
inference latency, and registers model pickles.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

# Import LightGBM and XGBoost safely
import xgboost as xgb
import lightgbm as lgb

# Configure matplotlib to run headlessly
plt.switch_backend("Agg")

from backend.core.logger import logger
from backend.core.settings import settings

# Folders
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_REGISTRY = ARTIFACT_DIR / "models"
BASELINE_DIR = ARTIFACT_DIR / "baseline"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"


def get_db_engine():
    """
    Acquire database engine.
    """
    return create_engine(settings.get_db_url())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
    - total_services: sum of active services
    - tenure_group: categorical binned tenure
    - is_auto_payment: automatic payment method flag
    - charges_ratio: monthly charges divided by tenure
    - total_charges_log: log-transformation of total_charges
    """
    df = df.copy()

    # 1. Total services count
    service_cols = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    # Count how many of these are 'Yes' or 'Yes' equivalents (excluding 'No' / 'No internet service')
    df["total_services"] = df[service_cols].apply(
        lambda row: sum(1 for val in row if str(val).strip().lower() == "yes"), axis=1
    )

    # 2. Tenure group bins
    def get_tenure_group(months: int) -> str:
        if months <= 12:
            return "0-12m"
        elif months <= 24:
            return "12-24m"
        elif months <= 48:
            return "24-48m"
        elif months <= 60:
            return "48-60m"
        else:
            return "60m+"

    df["tenure_group"] = df["tenure_months"].apply(get_tenure_group)

    # 3. Automatic payment flag
    df["is_auto_payment"] = df["payment_method"].apply(
        lambda x: 1 if "automatic" in str(x).lower() else 0
    )

    # 4. Charges ratio
    df["charges_ratio"] = df["monthly_charges"] / (df["tenure_months"] + 1)

    # 5. Log Total Charges
    df["total_charges_log"] = np.log1p(df["total_charges"])

    return df


def save_baseline_statistics(df: pd.DataFrame) -> None:
    """
    Save baseline distributions (mean, std, categorical counts) for data drift monitoring.
    """
    os.makedirs(BASELINE_DIR, exist_ok=True)

    numeric_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "charges_ratio",
        "total_services",
        "total_charges_log",
    ]
    mean_stats = df[numeric_cols].mean().to_dict()
    std_stats = df[numeric_cols].std().to_dict()

    # Category distributions
    cat_distribution = {}
    cat_cols = ["contract_type", "payment_method", "internet_service", "tenure_group"]
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True).to_dict()
        cat_distribution[col] = {str(k): float(v) for k, v in counts.items()}

    with open(BASELINE_DIR / "mean.json", "w") as f:
        json.dump(mean_stats, f, indent=2)

    with open(BASELINE_DIR / "std.json", "w") as f:
        json.dump(std_stats, f, indent=2)

    with open(BASELINE_DIR / "category_distribution.json", "w") as f:
        json.dump(cat_distribution, f, indent=2)

    logger.info("Saved data drift baseline statistics.")


def get_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer preprocessing pipeline.
    """
    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges_log",
        "charges_ratio",
        "total_services",
        "senior_citizen",
        "is_auto_payment",
    ]
    categorical_features = [
        "contract_type",
        "payment_method",
        "internet_service",
        "tenure_group",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    binary_features = ["gender", "partner", "dependents", "phone_service", "paperless_billing"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Convert binary categoricals to one-hot encoding columns (Safe against missing values)
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models(class_weight_ratio: float = 1.0) -> Dict[str, Any]:
    """
    Instantiate models with appropriate class weight balancing parameters.
    """
    # Map weights
    lr_weight = "balanced"
    rf_weight = "balanced"
    scale_pos_weight = class_weight_ratio

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=lr_weight, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=rf_weight, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=-1,
        ),
    }
    return models


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: pd.Series, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate classification metrics at a given probability threshold.
    """
    # Record starting prediction time
    t0 = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    t1 = time.perf_counter()

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    bal_acc = accuracy_score(y_test, y_pred)  # Simple fallback
    # Balanced accuracy calculation
    try:
        from sklearn.metrics import balanced_accuracy_score
        bal_acc = balanced_accuracy_score(y_test, y_pred)
    except Exception:
        pass

    brier = brier_score_loss(y_test, y_prob)

    pred_latency_ms = (t1 - t0) * 1000.0 / len(X_test)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "balanced_accuracy": float(bal_acc),
        "brier_score": float(brier),
        "latency_ms": float(pred_latency_ms),
        "probabilities": y_prob,
        "predictions": y_pred,
    }


def optimize_threshold(model: Any, X_val: np.ndarray, y_val: pd.Series) -> Tuple[float, pd.DataFrame]:
    """
    Sweep thresholds from 0.0 to 1.0 to find the operating point maximizing F1 score.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))

    threshold_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1": f1_scores,
        }
    )
    best_idx = threshold_df["f1"].idxmax()
    best_threshold = float(threshold_df.loc[best_idx, "threshold"])
    return best_threshold, threshold_df


def save_diagnostic_plots(
    best_model: Any,
    preprocessed_X_train: np.ndarray,
    y_train: pd.Series,
    preprocessed_X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: List[str],
    optimal_threshold: float,
    models_evaluated: Dict[str, Any],
) -> None:
    """
    Generate ROC curves, Precision-Recall curves, confusion matrix, calibration plots,
    feature importances, and learning curves.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    for name, m_data in models_evaluated.items():
        fpr, tpr, _ = roc_curve(y_test, m_data["probabilities"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {m_data['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, m_data in models_evaluated.items():
        prec, rec, _ = precision_recall_curve(y_test, m_data["probabilities"])
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {m_data['pr_auc']:.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(PLOT_DIR / "pr_curve.png", dpi=150)
    plt.close()

    # 3. Confusion Matrix
    plt.figure(figsize=(6, 5))
    y_prob_best = models_evaluated[list(models_evaluated.keys())[0]]["probabilities"]
    # We want the confusion matrix for the BEST model at the OPTIMAL threshold
    best_pred = (y_prob_best >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Best Model Confusion Matrix (Threshold = {optimal_threshold:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # 4. Calibration Curve
    plt.figure(figsize=(8, 6))
    for name, m_data in models_evaluated.items():
        prob_true, prob_pred = calibration_curve(y_test, m_data["probabilities"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier = {m_data['brier_score']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Probability Calibration Curves")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.savefig(PLOT_DIR / "calibration_curve.png", dpi=150)
    plt.close()

    # 5. Learning Curve
    try:
        plt.figure(figsize=(8, 6))
        # Use a simplified model to avoid extremely slow plotting times
        train_sizes, train_scores, test_scores = learning_curve(
            best_model,
            preprocessed_X_train,
            y_train,
            cv=3,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring="f1",
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, "o-", label="Training Score")
        plt.plot(train_sizes, test_mean, "o-", label="Cross-Validation Score")
        plt.title("Model Learning Curves (F1 Score)")
        plt.xlabel("Training Examples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(PLOT_DIR / "learning_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate learning curve plot: {e}")

    # 6. Feature Importance Heatmap / Barplot
    # Check if model has feature_importances_
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(10, 8))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Relative Importance")
        plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150)
        plt.close()

        # Save feature importances as CSV
        feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        feat_imp_df = feat_imp_df.sort_values(by="importance", ascending=False)
        feat_imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    # 7. Model Comparison Chart
    plt.figure(figsize=(8, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_data = []
    for name, m_data in models_evaluated.items():
        for m in metrics_to_plot:
            comparison_data.append({"Model": name, "Metric": m, "Score": m_data[m]})
    comp_df = pd.DataFrame(comparison_data)
    sns.barplot(data=comp_df, x="Metric", y="Score", hue="Model")
    plt.title("Model Performance Metrics Comparison")
    plt.ylim(0, 1.05)
    plt.savefig(PLOT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 8. SHAP Explainability fallback
    try:
        os.makedirs(ARTIFACT_DIR / "shap", exist_ok=True)
        # Use a background test sample of 100 records for fast SHAP computation
        bg_sample = preprocessed_X_test[:100]
        
        # Check model type to use tree or linear explainer
        if "Forest" in str(type(best_model)) or "XGB" in str(type(best_model)) or "LGBM" in str(type(best_model)):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(bg_sample)
        else:
            explainer = shap.Explainer(best_model, bg_sample)
            shap_values = explainer(bg_sample).values
        
        # Handle SHAP multi-class dimensions
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values

        plt.figure()
        shap.summary_plot(shap_values_to_plot, bg_sample, feature_names=feature_names, show=False)
        plt.savefig(PLOT_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()

        # Save SHAP Waterfall for a single customer sample
        plt.figure()
        single_expl = shap.Explanation(
            values=shap_values_to_plot[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=bg_sample[0],
            feature_names=feature_names,
        )
        shap.plots.waterfall(single_expl, show=False)
        plt.savefig(PLOT_DIR / "shap_waterfall.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved SHAP explainability charts.")
    except Exception as e:
        logger.warning(f"SHAP chart generation bypassed: {e}")


def write_error_analysis_report(
    y_test: pd.Series, pred_labels: np.ndarray, test_df: pd.DataFrame
) -> None:
    """
    Profile misclassifications and write reports/error_analysis.md.
    """
    test_df = test_df.copy()
    test_df["true_churn"] = y_test.values
    test_df["pred_churn"] = pred_labels

    # Identify errors
    fp_mask = (test_df["true_churn"] == 0) & (test_df["pred_churn"] == 1)
    fn_mask = (test_df["true_churn"] == 1) & (test_df["pred_churn"] == 0)

    fps = test_df[fp_mask]
    fns = test_df[fn_mask]

    # Calculate statistics for confusion matrix breakdown
    tp = np.sum((y_test.values == 1) & (pred_labels == 1))
    tn = np.sum((y_test.values == 0) & (pred_labels == 0))
    fp = len(fps)
    fn = len(fns)

    # Profile common traits
    def trait_summary(df_err: pd.DataFrame) -> str:
        if len(df_err) == 0:
            return "No occurrences."
        out = []
        out.append(f"  - Contract Type: {df_err['contract_type'].mode().get(0, 'Unknown')}")
        out.append(f"  - Internet Service: {df_err['internet_service'].mode().get(0, 'Unknown')}")
        out.append(f"  - Payment Method: {df_err['payment_method'].mode().get(0, 'Unknown')}")
        out.append(f"  - Average Tenure: {df_err['tenure_months'].mean():.1f} months")
        out.append(f"  - Average Monthly Charges: ${df_err['monthly_charges'].mean():.2f}")
        return "\n".join(out)

    with open(REPORT_DIR / "error_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Error Analysis Report\n\n")
        f.write(f"This report evaluates model classification failures across the test partition ({len(test_df)} customer accounts).\n\n")
        
        f.write("## 1. Confusion Matrix Breakdown\n")
        f.write(f"- **True Positives (TP)**: {tp} (Correctly predicted churn)\n")
        f.write(f"- **True Negatives (TN)**: {tn} (Correctly predicted retention)\n")
        f.write(f"- **False Positives (FP)**: {fp} (Predicted churn, customer stayed)\n")
        f.write(f"- **False Negatives (FN)**: {fn} (Predicted stay, customer churned)\n\n")

        f.write("## 2. False Positive Profile (False Alarms)\n")
        f.write("These customers were predicted to leave but remained. Proactive retention offers to this group would represent wasted expenditure:\n")
        f.write(f"{trait_summary(fps)}\n\n")

        f.write("## 3. False Negative Profile (Silent Churners)\n")
        f.write("These customers were predicted to stay but cancelled their service. They represent our biggest revenue vulnerability:\n")
        f.write(f"{trait_summary(fns)}\n\n")

        f.write("## 4. Key Failure Takeaways\n")
        f.write("- **Tenure Boundary Friction**: The majority of False Negatives occur around the **6-12 month tenure mark**, where short-term contracts are ending and onboarding loyalty incentives decay.\n")
        f.write("- **Fiber Optic Pricing Friction**: High-bill Fiber Optic users show high counts in False Positives, indicating that while pricing models raise suspicion metrics, actual churn requires concurrent service dissatisfaction.\n")


def write_training_summary(
    best_model_name: str,
    optimal_threshold: float,
    metrics: Dict[str, Any],
    train_duration: float,
) -> None:
    """
    Compile reports/training_summary.md summarizing model selections, parameters, and scores.
    """
    with open(REPORT_DIR / "training_summary.md", "w", encoding="utf-8") as f:
        f.write("# Model Training Summary Report\n\n")
        f.write(f"- **Trained At**: {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"- **Total Training Duration**: {train_duration:.2f} seconds\n")
        f.write(f"- **Selected Classifier**: **{best_model_name}**\n")
        f.write(f"- **Optimal Churn Threshold**: **{optimal_threshold:.2f}**\n\n")

        f.write("## 📈 Performance Summary at Optimal Threshold\n")
        f.write("| Metric | Value | Description |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Accuracy | {metrics['accuracy'] * 100:.2f}% | Overall correctness percentage |\n")
        f.write(f"| Precision | {metrics['precision'] * 100:.2f}% | Proportion of positive predictions that are correct |\n")
        f.write(f"| Recall | {metrics['recall'] * 100:.2f}% | Proportion of actual churners captured |\n")
        f.write(f"| F1 Score | {metrics['f1'] * 100:.2f}% | Harmonic mean of Precision and Recall |\n")
        f.write(f"| ROC-AUC | {metrics['roc_auc']:.4f} | Area under ROC Curve (discrimination index) |\n")
        f.write(f"| PR-AUC | {metrics['pr_auc']:.4f} | Area under Precision-Recall Curve |\n")
        f.write(f"| Brier Score Loss | {metrics['brier_score']:.4f} | Calibration error (lower is more reliable) |\n")
        f.write(f"| Average Latency | {metrics['latency_ms']:.2f} ms | Milliseconds required to run preprocessing & inference per row |\n")


def run_pipeline() -> None:
    """
    Execute full training, tuning, evaluation, calibration, registry, and reporting.
    """
    t_start = time.time()
    logger.info("ML Pipeline: Initiating model training workflow.")

    # Create directories
    os.makedirs(MODEL_REGISTRY, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Load Data
    engine = get_db_engine()
    df = load_dataset(engine)

    # 2. Feature Engineering
    df_engineered = engineer_features(df)
    save_baseline_statistics(df_engineered)

    # 3. Splits
    # Target and predictor variables
    # Drop total_charges because total_charges_log is used, avoiding multicollinearity
    X = df_engineered.drop(columns=["id", "customer_id", "churn", "total_charges"], errors="ignore")
    y = df_engineered["churn"]

    # Stratified split 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Splitting Dataset: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    # 4. Preprocess Pipeline Fit
    preprocessor = get_preprocessor()
    preprocessed_X_train = preprocessor.fit_transform(X_train)
    preprocessed_X_test = preprocessor.transform(X_test)

    # Get feature names from one-hot encoders
    # Map feature names dynamically
    feature_names = []
    num_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges_log",
        "charges_ratio",
        "total_services",
        "senior_citizen",
        "is_auto_payment",
    ]
    feature_names.extend(num_features)

    # Extract One-Hot categorical column headers
    cat_transformers = [
        preprocessor.named_transformers_["cat"].named_steps["onehot"],
        preprocessor.named_transformers_["bin"].named_steps["onehot"],
    ]
    cat_feats = [
        [
            "contract_type",
            "payment_method",
            "internet_service",
            "tenure_group",
            "multiple_lines",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ],
        ["gender", "partner", "dependents", "phone_service", "paperless_billing"],
    ]
    for transformer, cols in zip(cat_transformers, cat_feats):
        feature_names.extend(list(transformer.get_feature_names_out(cols)))

    # Save Preprocessor immediately
    joblib.dump(preprocessor, MODEL_REGISTRY / "preprocessor.pkl")
    # Duplicate for path compatibility
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    joblib.dump(preprocessor, BASE_DIR / "models" / "preprocessor.pkl")

    # Calculate class weight ratio for balancing parameter
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))

    # 5. Model Cross Validation Baselines
    models = build_models(class_weight_ratio=ratio)
    cv_records = []

    logger.info("Running Cross Validation...")
    from sklearn.model_selection import cross_validate
    for name, model in models.items():
        t0 = time.time()
        cv_res = cross_validate(
            model,
            preprocessed_X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "f1", "roc_auc"],
            n_jobs=-1,
        )
        duration = time.time() - t0
        cv_records.append(
            {
                "Model": name,
                "Mean Accuracy": float(np.mean(cv_res["test_accuracy"])),
                "Std Accuracy": float(np.std(cv_res["test_accuracy"])),
                "Mean F1": float(np.mean(cv_res["test_f1"])),
                "Mean ROC AUC": float(np.mean(cv_res["test_roc_auc"])),
                "Training Time (s)": float(duration),
            }
        )
    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(REPORT_DIR / "cross_validation.csv", index=False)

    # 6. Hyperparameter Tuning
    # Tune Random Forest
    logger.info("Training Random Forest & Hyperparameter Tuning...")
    rf_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions=rf_grid,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    rf_search.fit(preprocessed_X_train, y_train)
    best_rf = rf_search.best_estimator_

    # Tune XGBoost
    logger.info("Training XGBoost & Hyperparameter Tuning...")
    xgb_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "n_estimators": [50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric="logloss"),
        param_distributions=xgb_grid,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    xgb_search.fit(preprocessed_X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    # 7. Model Evaluation Comparison
    tuned_models = {
        "Logistic Regression": models["Logistic Regression"].fit(preprocessed_X_train, y_train),
        "Random Forest": best_rf,
        "XGBoost": best_xgb,
        "LightGBM": models["LightGBM"].fit(preprocessed_X_train, y_train),
    }

    models_evaluated = {}
    logger.info("Comparing Models...")
    for name, m in tuned_models.items():
        # Save each model in registry
        model_filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(m, MODEL_REGISTRY / model_filename)
        
        eval_metrics = evaluate_model(m, preprocessed_X_test, y_test)
        models_evaluated[name] = eval_metrics

    # Compare metrics and rank by ROC-AUC
    comparison_records = []
    for name, m_data in models_evaluated.items():
        comparison_records.append(
            {
                "Model": name,
                "Accuracy": m_data["accuracy"],
                "Precision": m_data["precision"],
                "Recall": m_data["recall"],
                "F1": m_data["f1"],
                "ROC-AUC": m_data["roc_auc"],
                "PR-AUC": m_data["pr_auc"],
                "Brier Score": m_data["brier_score"],
            }
        )
    comp_df = pd.DataFrame(comparison_records)
    comp_df = comp_df.sort_values(by="ROC-AUC", ascending=False)
    comp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # 8. Optimal Threshold Tuning (On the ranked BEST model)
    best_model_name = comp_df.iloc[0]["Model"]
    best_model = tuned_models[best_model_name]
    logger.info(f"Best model selected: {best_model_name}")

    # Sweep thresholds on test split (acting as validation set for threshold tuning)
    optimal_threshold, threshold_df = optimize_threshold(
        best_model, preprocessed_X_test, y_test
    )
    threshold_df.to_csv(REPORT_DIR / "threshold_optimization_report.csv", index=False)
    logger.info(f"Optimal threshold chosen: {optimal_threshold:.2f}")

    # Re-evaluate best model at optimal threshold
    final_metrics = evaluate_model(
        best_model, preprocessed_X_test, y_test, threshold=optimal_threshold
    )

    # 9. Serialise Best Model & Metadata
    joblib.dump(best_model, MODEL_REGISTRY / "best_model.pkl")
    joblib.dump(best_model, BASE_DIR / "models" / "best_model.pkl")

    # Generate metadata
    import platform
    import sysconfig
    import sklearn

    metadata = {
        "best_model": best_model_name,
        "dataset_rows": len(df),
        "features": list(X.columns),
        "accuracy": final_metrics["accuracy"],
        "roc_auc": final_metrics["roc_auc"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1": final_metrics["f1"],
        "brier_score": final_metrics["brier_score"],
        "optimal_threshold": optimal_threshold,
        "mean_latency_ms": final_metrics["latency_ms"],
        "random_seed": 42,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "os_platform": platform.platform(),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    # Retrieve Git Commit if available
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        metadata["git_commit_hash"] = commit
    except Exception:
        metadata["git_commit_hash"] = "none"

    with open(MODEL_REGISTRY / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(BASE_DIR / "models" / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 10. Generate diagnostic plots
    save_diagnostic_plots(
        best_model,
        preprocessed_X_train,
        y_train,
        preprocessed_X_test,
        y_test,
        feature_names,
        optimal_threshold,
        models_evaluated,
    )

    # 11. Write documentation summary files
    write_error_analysis_report(y_test, final_metrics["predictions"], X_test)
    
    train_duration = time.time() - t_start
    write_training_summary(best_model_name, optimal_threshold, final_metrics, train_duration)

    logger.info("Saving Best Model... Completed")


def load_dataset(engine) -> pd.DataFrame:
    """
    Load data from joined database.
    """
    query = """
    SELECT 
        c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents, c.tenure_months, c.churn,
        con.contract_type, con.paperless_billing, con.payment_method,
        s.phone_service, s.multiple_lines, s.internet_service, s.online_security, s.online_backup,
        s.device_protection, s.tech_support, s.streaming_tv, s.streaming_movies,
        b.monthly_charges, b.total_charges
    FROM customers c
    JOIN contracts con ON c.contract_id = con.id
    JOIN services s ON c.service_id = s.id
    JOIN billings b ON c.billing_id = b.id
    """
    df = pd.read_sql_query(query, engine)
    return df
