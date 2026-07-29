"""
Telecom Customer Intelligence Engine & Batch Training Script.

Executes end-to-end Telecom Subscriber Intelligence calculations:
  1. Train / Predict Churn probability (LGBM)
  2. Train LTV regressors & estimate subscriber lifetime value
  3. Run K-Means Telecom Subscriber Clustering (High-Value, Loyal, Growth, Budget)
  4. Perform Telecom RFM Analysis (VIP, Loyal, High-Potential, At-Risk, Churned, Dormant)
  5. Calculate Composite Subscriber Intelligence Scores (0-100)
  6. Execute Hybrid Retention Recommendation Engine
  7. Export analytics CSVs, PNG charts, and Markdown Reports.

Run:
    python scripts/train_intelligence.py
"""

import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ml.cleaning import DataCleaner
from backend.ml.training import engineer_features
from backend.ml.intelligence import (
    train_ltv_models,
    run_customer_segmentation,
    calculate_rfm,
    calculate_intelligence_score,
    generate_recommendation_details,
)
from backend.services.predict_service import PredictService
from backend.core.logger import logger

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"
MODEL_REGISTRY = BASE_DIR / "artifacts" / "models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_REGISTRY, exist_ok=True)


def main() -> None:
    logger.info("Telecom Intelligence Pipeline: Starting execution...")

    # 1. Load and clean baseline Telco dataset
    csv_path = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "processed" / "cleaned_telco.csv"
    if not csv_path.exists():
        csv_path = REPORT_DIR / "customer_intelligence.csv"

    df_raw = pd.read_csv(csv_path)
    if "TotalCharges" in df_raw.columns or "customerID" in df_raw.columns:
        cleaner = DataCleaner()
        df, _ = cleaner.clean(df_raw)
    else:
        df = df_raw.copy()
    logger.info(f"Loaded {len(df)} telecom subscriber records.")

    # 2. Score Churn Probabilities
    logger.info("Inference: Batch evaluating churn probabilities...")
    predict_service = PredictService()
    
    df_eng_churn = engineer_features(df)
    df_final_churn = df_eng_churn[predict_service.feature_columns]
    X_trans_churn = predict_service.preprocessor.transform(df_final_churn)
    churn_probs = predict_service.model.predict_proba(X_trans_churn)[:, 1].tolist()
    df["churn_probability"] = churn_probs

    # 3. Fit LTV Regression Models
    logger.info("ML Regressions: Training subscriber lifetime value regressors...")
    df_eng = engineer_features(df)
    ltv_pipeline, ltv_meta, ltv_comp_df = train_ltv_models(df_eng)

    # Serialize LTV Pipeline
    joblib.dump(ltv_pipeline, MODEL_REGISTRY / "ltv_model.pkl")
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    joblib.dump(ltv_pipeline, BASE_DIR / "models" / "ltv_model.pkl")

    with open(MODEL_REGISTRY / "ltv_metadata.json", "w") as f:
        json.dump(ltv_meta, f, indent=2)
    with open(BASE_DIR / "models" / "ltv_metadata.json", "w") as f:
        json.dump(ltv_meta, f, indent=2)

    ltv_comp_df.to_csv(REPORT_DIR / "ltv_model_comparison.csv", index=False)

    df_eng["predicted_ltv"] = ltv_pipeline.predict(df_eng)

    # Forecast Projected LTV
    expected_remaining_lifetime = (1.0 / df_eng["churn_probability"].clip(lower=0.01)) - df_eng["tenure_months"]
    expected_remaining_lifetime = expected_remaining_lifetime.clip(lower=0.0)
    df_eng["projected_future_ltv"] = expected_remaining_lifetime * df_eng["monthly_charges"]
    
    # 4. K-Means Telecom Subscriber Clustering
    logger.info("ML Clustering: Performing telecom subscriber segmentation...")
    seg_pipeline, stats_k_df, segments = run_customer_segmentation(df_eng)

    joblib.dump(seg_pipeline, MODEL_REGISTRY / "segmentation_model.pkl")
    joblib.dump(seg_pipeline, BASE_DIR / "models" / "segmentation_model.pkl")
    stats_k_df.to_csv(REPORT_DIR / "segment_statistics.csv", index=False)

    df_eng["customer_segment"] = segments.values

    # 5. Telecom RFM Analysis
    logger.info("RFM Analysis: Scoring subscriber usage patterns...")
    df_rfm = calculate_rfm(df_eng, df_eng["churn_probability"].values)
    df_rfm.to_csv(REPORT_DIR / "rfm_analysis.csv", index=False)
    
    df_eng["rfm_persona"] = df_rfm["persona"].values

    # 6. Composite Subscriber Intelligence Scoring (0-100)
    logger.info("Scoring: Computing unified Telecom Subscriber Intelligence Scores...")
    scores = []
    categories = []
    max_ltv = float(df_eng["predicted_ltv"].max())
    
    for _, row in df_eng.iterrows():
        sc, cat = calculate_intelligence_score(
            row["churn_probability"],
            row["predicted_ltv"],
            row["tenure_months"],
            row["total_services"],
            max_ltv=max_ltv
        )
        scores.append(sc)
        categories.append(cat)

    df_eng["intelligence_score"] = scores
    df_eng["intelligence_category"] = categories

    # 7. Telecom Recommendation Engine
    logger.info("Recommendations: Generating proactive retention recommendations...")
    primary_recs = []
    priorities = []
    revenues_saved = []

    for _, row in df_eng.iterrows():
        sample_dict = row.to_dict()
        shap_driver = ""
        if row["churn_probability"] >= 0.40:
            if row["contract_type"] == "Month-to-month":
                shap_driver = "Month-to-month contract structure"
            elif row["tenure_months"] <= 6:
                shap_driver = "Short subscriber tenure"
            else:
                shap_driver = "High monthly fee sensitivity"

        recs = generate_recommendation_details(
            sample_dict,
            row["churn_probability"],
            row["predicted_ltv"],
            row["customer_segment"],
            row["rfm_persona"],
            shap_top_contrib=shap_driver
        )
        primary_recs.append(recs[0]["recommendation"])
        priorities.append(recs[0]["priority"])
        revenues_saved.append(recs[0]["estimated_revenue_saved"])

    df_eng["primary_recommendation"] = primary_recs
    df_eng["recommendation_priority"] = priorities
    df_eng["estimated_revenue_saved"] = revenues_saved

    # 8. Save Unified Telecom Analytics File
    logger.info("Saving: Exporting telecom analytics files to disk...")
    export_cols = [
        "customer_id", "churn_probability", "predicted_ltv", "projected_future_ltv",
        "customer_segment", "rfm_persona", "intelligence_score", "intelligence_category",
        "primary_recommendation", "recommendation_priority", "estimated_revenue_saved"
    ]
    df_intelligence = df_eng[export_cols]
    df_intelligence.to_csv(REPORT_DIR / "customer_intelligence.csv", index=False)
    
    df_segments = df_eng[["customer_id", "customer_segment"]]
    df_segments.to_csv(REPORT_DIR / "customer_segments.csv", index=False)

    # 9. Recommendation Analytics
    rec_counts = df_eng["primary_recommendation"].value_counts().to_frame("subscribers")
    rec_counts.to_csv(REPORT_DIR / "recommendation_statistics.csv")

    # 10. Generate Visual Analytics Charts
    sns.set_theme(style="whitegrid", palette="muted")

    # 10.1 LTV Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_eng, x="predicted_ltv", kde=True, bins=30, color="purple")
    plt.title("Subscriber Lifetime Value (LTV Proxy) Distribution")
    plt.xlabel("LTV ($)")
    plt.savefig(PLOT_DIR / "ltv_distribution.png", dpi=150)
    plt.close()

    # 10.2 Subscriber Segments
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df_eng,
        x="customer_segment",
        order=["High-Value Subscribers", "Loyal Subscribers", "Growth Subscribers", "Budget Subscribers"]
    )
    plt.title("Telecom Subscriber Segments Distribution")
    plt.xlabel("Subscriber Segment")
    plt.xticks(rotation=15)
    plt.savefig(PLOT_DIR / "segment_distribution.png", dpi=150)
    plt.close()

    # 10.3 Cluster Visualisation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_eng, x="tenure_months", y="monthly_charges", hue="customer_segment", alpha=0.6)
    plt.title("Telecom Segmentation Clusters (Tenure vs Monthly Charges)")
    plt.savefig(PLOT_DIR / "cluster_visualization.png", dpi=150)
    plt.close()

    # 10.4 RFM Heatmap
    plt.figure(figsize=(8, 6))
    rfm_matrix = df_rfm.groupby(["R_score", "F_score"])["M_score"].mean().unstack().fillna(0)
    sns.heatmap(rfm_matrix, annot=True, cmap="Purples", fmt=".2f")
    plt.title("Telecom RFM Heatmap: Mean Monetary Score by Recency & Frequency")
    plt.xlabel("Frequency Score")
    plt.ylabel("Recency Score")
    plt.savefig(PLOT_DIR / "rfm_heatmap.png", dpi=150)
    plt.close()

    # 10.5 Recommendation Chart
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_eng, y="primary_recommendation", hue="recommendation_priority")
    plt.title("Telecom Retention Offers Priority Distribution")
    plt.xlabel("Subscriber Count")
    plt.savefig(PLOT_DIR / "recommendation_chart.png", dpi=150)
    plt.close()

    # 10.6 Revenue Distribution
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        data=df_eng,
        x="customer_segment",
        y="total_charges",
        order=["High-Value Subscribers", "Loyal Subscribers", "Growth Subscribers", "Budget Subscribers"]
    )
    plt.title("Accumulated Revenue Distribution by Telecom Segment")
    plt.xticks(rotation=15)
    plt.savefig(PLOT_DIR / "revenue_distribution.png", dpi=150)
    plt.close()

    # 10.7 Segment Comparison
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_eng,
        x="customer_segment",
        y="churn_probability",
        order=["High-Value Subscribers", "Loyal Subscribers", "Growth Subscribers", "Budget Subscribers"]
    )
    plt.title("Churn Probability distribution by Telecom Segment")
    plt.xticks(rotation=15)
    plt.savefig(PLOT_DIR / "segment_comparison.png", dpi=150)
    plt.close()

    # 10.8 Score Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_eng, x="intelligence_score", kde=True, bins=30, color="green")
    plt.title("Telecom Subscriber Intelligence Score (0-100) Distribution")
    plt.xlabel("Subscriber Score")
    plt.savefig(PLOT_DIR / "score_distribution.png", dpi=150)
    plt.close()

    # 10.9 Top Revenue Subscribers
    plt.figure(figsize=(8, 4))
    top_rev = df_eng.sort_values(by="total_charges", ascending=False).head(10)
    sns.barplot(data=top_rev, x="total_charges", y="customer_id", color="gold")
    plt.title("Top 10 Accounts by Accumulated Subscriber Spend")
    plt.xlabel("Revenue ($)")
    plt.savefig(PLOT_DIR / "top_revenue_customers.png", dpi=150)
    plt.close()

    # 10.10 Business Impact Dashboard Plot
    plt.figure(figsize=(8, 5))
    impact_data = df_eng.groupby("recommendation_priority")["estimated_revenue_saved"].sum().reset_index()
    sns.barplot(data=impact_data, x="recommendation_priority", y="estimated_revenue_saved", palette="viridis")
    plt.title("Estimated Potential Revenue Saved by Campaign Priority")
    plt.ylabel("Revenue Saved ($)")
    plt.savefig(PLOT_DIR / "business_dashboard.png", dpi=150)
    plt.close()

    # 11. Write Markdown Reports
    write_reports(df_eng, ltv_meta, stats_k_df)

    logger.info("Intelligence Pipeline: Completed successfully.")


def write_reports(df_eng: pd.DataFrame, ltv_meta: Dict[str, Any], stats_k_df: pd.DataFrame) -> None:
    """
    Generate the markdown intelligence reports using telecom business terminology.
    """
    seg_counts = df_eng["customer_segment"].value_counts().to_dict()
    persona_counts = df_eng["rfm_persona"].value_counts().to_dict()
    total_rev_saved = df_eng["estimated_revenue_saved"].sum()

    # 1. ltv_summary.md
    with open(REPORT_DIR / "ltv_summary.md", "w", encoding="utf-8") as f:
        f.write("# Subscriber Lifetime Value (LTV) Prediction Summary\n\n")
        f.write("This report documents the LTV prediction regression performance and future subscriber forecasts.\n\n")
        f.write("## 1. Regression Model Comparison (Historical Proxy)\n")
        f.write(f"- **Selected Best Model**: {ltv_meta['best_model']}\n")
        f.write(f"- **R² Coefficient of Determination**: {ltv_meta['r2']:.4f}\n")
        f.write(f"- **Root Mean Squared Error (RMSE)**: ${ltv_meta['rmse']:.2f}\n")
        f.write(f"- **Mean Absolute Error (MAE)**: ${ltv_meta['mae']:.2f}\n")
        f.write(f"- **Mean Absolute Percentage Error (MAPE)**: {ltv_meta['mape']:.2f}%\n\n")
        f.write("## 2. Hybrid LTV Target Definition\n")
        f.write("We use `total_charges` as a historical proxy of accumulated spend. We then estimate the **Projected Future LTV** via expected remaining contract lifetime:\n")
        f.write("$$\\text{Remaining Lifetime (months)} = \\max\\left(0, \\frac{1}{\\text{Churn Probability}} - \\text{Tenure}\\right)$$\n")
        f.write("$$\\text{Projected Future LTV} = \\text{Remaining Lifetime} \\times \\text{Monthly Charges}$$\n\n")
        f.write(f"- **Total Projected Future LTV Potential**: ${df_eng['projected_future_ltv'].sum():,.2f}\n")
        f.write(f"- **Average Projected Future LTV**: ${df_eng['projected_future_ltv'].mean():.2f} per subscriber\n")

    # 2. segment_profiles.md
    with open(REPORT_DIR / "segment_profiles.md", "w", encoding="utf-8") as f:
        f.write("# Telecom Subscriber Segmentation Profiles Report\n\n")
        f.write("This report outlines the characteristics of the telecom subscriber clusters generated via K-Means.\n\n")
        
        segments_list = ["High-Value Subscribers", "Loyal Subscribers", "Growth Subscribers", "Budget Subscribers"]
        for seg in segments_list:
            df_seg = df_eng[df_eng["customer_segment"] == seg]
            if len(df_seg) == 0:
                continue
            
            f.write(f"## 🏆 {seg} Profile\n")
            f.write(f"- **Subscriber Count**: {len(df_seg)} ({len(df_seg)/len(df_eng)*100:.1f}%)\n")
            f.write(f"- **Average Historical LTV**: ${df_seg['predicted_ltv'].mean():.2f}\n")
            f.write(f"- **Average Monthly Charges**: ${df_seg['monthly_charges'].mean():.2f}\n")
            f.write(f"- **Average Tenure**: {df_seg['tenure_months'].mean():.1f} months\n")
            f.write(f"- **Average Churn Risk**: {df_seg['churn_probability'].mean()*100:.1f}%\n")
            
            top_contract = df_seg["contract_type"].mode().get(0, "Unknown")
            top_payment = df_seg["payment_method"].mode().get(0, "Unknown")
            f.write(f"- **Primary Contract Type**: {top_contract}\n")
            f.write(f"- **Primary Payment Method**: {top_payment}\n")
            
            campaign = "Standard Loyalty Campaign"
            if seg == "High-Value Subscribers":
                campaign = "Offer Executive Retention & Fiber Upgrade Pack"
            elif seg == "Loyal Subscribers":
                campaign = "Offer Multi-Service OTT & Security Bundle"
            elif seg == "Growth Subscribers":
                campaign = "Offer Annual Contract Migration Discount"
            elif seg == "Budget Subscribers":
                campaign = "Offer Autopay Billing Discount & Entry Fiber Upgrade"
                
            f.write(f"- **Recommended Retention Offer**: {campaign}\n\n")

    # 3. executive_summary.md
    with open(REPORT_DIR / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write("# Executive Telecom Intelligence Summary\n\n")
        f.write(f"- **Date of Report**: {datetime.utcnow().date().isoformat()}\n")
        f.write(f"- **Total Active Subscribers Analyzed**: {len(df_eng)}\n")
        f.write(f"- **Overall Database Churn Rate**: {df_eng['churn'].mean()*100:.1f}%\n")
        f.write(f"- **Overall Projected Churn Risk (Probabilistic)**: {df_eng['churn_probability'].mean()*100:.1f}%\n\n")
        
        f.write("## 1. Telecom Segment Share\n")
        for k, v in seg_counts.items():
            f.write(f"- **{k}**: {v} subscribers ({v/len(df_eng)*100:.1f}%)\n")
            
        f.write("\n## 2. Telecom RFM Personas Share\n")
        for k, v in list(persona_counts.items())[:5]:
            f.write(f"- **{k}**: {v} subscribers ({v/len(df_eng)*100:.1f}%)\n")
            
        f.write("\n## 3. High-Risk Retention Actionables\n")
        critical_count = len(df_eng[df_eng["recommendation_priority"] == "Critical"])
        high_count = len(df_eng[df_eng["recommendation_priority"] == "High"])
        f.write(f"- **Critical Priority Actions Required**: {critical_count} subscribers\n")
        f.write(f"- **High Priority Actions Required**: {high_count} subscribers\n")

    # 4. business_impact.md
    with open(REPORT_DIR / "business_impact.md", "w", encoding="utf-8") as f:
        f.write("# Business Impact & Telecom Financial Simulation\n\n")
        f.write("We estimate the financial savings from acting on telecom recommendation triggers.\n\n")
        f.write(f"- **Total Revenue at Attrition Risk (Probabilistic)**: ${df_eng['churn_probability'].dot(df_eng['total_charges']):,.2f}\n")
        f.write(f"- **Maximum Potential Revenue Saved**: ${total_rev_saved:,.2f}\n\n")
        f.write("## 1. Financial Impact Simulation (50% Response Assumption)\n")
        f.write("If **50%** of targeted high-risk subscribers accept our proactive retention/upgrade recommendations:\n")
        f.write(f"- **Targeted Subscribers**: {len(df_eng[df_eng['churn_probability'] >= 0.40])} accounts\n")
        f.write(f"- **Estimated Financial Savings**: **${total_rev_saved * 0.50:,.2f}**\n")

    # 5. recommendation_summary.md
    with open(REPORT_DIR / "recommendation_summary.md", "w", encoding="utf-8") as f:
        f.write("# Retention Recommendation Engine Output Summary\n\n")
        f.write("This report summarizes the volumes and priority of all triggered telecom retention offers.\n\n")
        recs_summary = df_eng.groupby(["primary_recommendation", "recommendation_priority"])["customer_id"].count().reset_index()
        f.write("| Retention Recommendation Offer | Priority | Targeted Subscribers | Expected Saved Rev |\n")
        f.write("| --- | --- | --- | --- |\n")
        for _, row in recs_summary.iterrows():
            sub_df = df_eng[df_eng["primary_recommendation"] == row["primary_recommendation"]]
            rev_sum = sub_df["estimated_revenue_saved"].sum()
            f.write(f"| {row['primary_recommendation']} | {row['recommendation_priority']} | {row['customer_id']} | ${rev_sum:,.2f} |\n")

    # 6. customer_intelligence.md
    with open(REPORT_DIR / "customer_intelligence.md", "w", encoding="utf-8") as f:
        f.write("# Unified Telecom Subscriber Intelligence Scoring Methodology\n\n")
        f.write("The Unified **Subscriber Intelligence Score (0-100)** is computed as a weighted index of value and risk metrics:\n\n")
        f.write("## 1. Scoring Formula\n")
        f.write("```text\n")
        f.write("Score = 30% * (1.0 - Churn Probability) * 100\n")
        f.write("      + 30% * (log(Predicted LTV) / log(Max LTV)) * 100\n")
        f.write("      + 20% * (Tenure / 72 months) * 100\n")
        f.write("      + 20% * (Services / 8 services) * 100\n")
        f.write("```\n")


if __name__ == "__main__":
    main()
