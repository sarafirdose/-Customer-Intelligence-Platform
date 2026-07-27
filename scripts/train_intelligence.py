"""
Customer Intelligence Platform - Pipeline Training Runner.

Loads dataset, gets churn predictions from classification model, trains LTV regressors
and K-Means clusters, performs RFM analyses and scoring, generates visual charts
and markdown reports, and registers intelligence models.
"""

import json
import joblib
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from typing import Any, Dict, List

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import logger
from backend.core.settings import settings
from backend.ml.training import engineer_features
from backend.services.predict_service import PredictService
from backend.ml.intelligence import (
    train_ltv_models,
    run_customer_segmentation,
    calculate_rfm,
    calculate_intelligence_score,
    generate_recommendation_details,
)

# Output paths
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_REGISTRY = ARTIFACT_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

# Configure matplotlib to run headlessly
plt.switch_backend("Agg")


def get_db_engine():
    """
    Acquire SQLite/Postgres database engine.
    """
    return create_engine(settings.get_db_url())


def main() -> None:
    """
    Execute training and analytics calculations.
    """
    logger.info("Intelligence Pipeline: Starting customer intelligence calculations.")

    # 1. Load Data
    engine = get_db_engine()
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
    logger.info(f"Loaded {len(df)} customer records.")

    # 2. Extract Churn Probabilities from Phase 3 Churn Model
    logger.info("Inference: Batch evaluating churn probabilities...")
    predict_service = PredictService()
    
    # Run batch feature engineering and transform
    df_eng_churn = engineer_features(df)
    df_final_churn = df_eng_churn[predict_service.feature_columns]
    X_trans_churn = predict_service.preprocessor.transform(df_final_churn)
    churn_probs = predict_service.model.predict_proba(X_trans_churn)[:, 1].tolist()
    df["churn_probability"] = churn_probs

    # 3. Fit LTV Regression Models
    logger.info("ML Regressions: Training customer lifetime value regressors...")
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

    # Predict LTV values (total_charges historical proxy)
    df_eng["predicted_ltv"] = ltv_pipeline.predict(df_eng)

    # Projected LTV forecast based on Churn lifespans
    expected_remaining_lifetime = (1.0 / df_eng["churn_probability"].clip(lower=0.01)) - df_eng["tenure_months"]
    expected_remaining_lifetime = expected_remaining_lifetime.clip(lower=0.0)
    df_eng["projected_future_ltv"] = expected_remaining_lifetime * df_eng["monthly_charges"]
    
    # 4. K-Means Customer Clustering Segmentation
    logger.info("ML Clustering: Performing customer segmentation...")
    seg_pipeline, stats_k_df, segments = run_customer_segmentation(df_eng)

    # Serialize Segmentation
    joblib.dump(seg_pipeline, MODEL_REGISTRY / "segmentation_model.pkl")
    joblib.dump(seg_pipeline, BASE_DIR / "models" / "segmentation_model.pkl")
    stats_k_df.to_csv(REPORT_DIR / "segment_statistics.csv", index=False)

    df_eng["customer_segment"] = segments.values

    # 5. RFM Analysis
    logger.info("RFM Analysis: Scoring customer transaction patterns...")
    df_rfm = calculate_rfm(df_eng, df_eng["churn_probability"].values)
    df_rfm.to_csv(REPORT_DIR / "rfm_analysis.csv", index=False)
    
    df_eng["rfm_persona"] = df_rfm["persona"].values

    # 6. Customer Intelligence Scoring (0-100)
    logger.info("Scoring: Computing unified Customer Intelligence Scores...")
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

    # 7. Hybrid Recommendation Engine Logic
    logger.info("Recommendations: Generating rules retention recommendations...")
    primary_recs = []
    priorities = []
    revenues_saved = []

    for _, row in df_eng.iterrows():
        sample_dict = row.to_dict()
        # Mock SHAP top driver contribution if high churn risk
        shap_driver = ""
        if row["churn_probability"] >= 0.40:
            # Check largest feature values to highlight
            if row["contract_type"] == "Month-to-month":
                shap_driver = "Month-to-month contract structure"
            elif row["tenure_months"] <= 6:
                shap_driver = "Very short onboarding tenure"
            else:
                shap_driver = "High monthly charges fee sensitivity"

        recs = generate_recommendation_details(
            sample_dict,
            row["churn_probability"],
            row["predicted_ltv"],
            row["customer_segment"],
            row["rfm_persona"],
            shap_top_contrib=shap_driver
        )
        # Record primary recommendation
        primary_recs.append(recs[0]["recommendation"])
        priorities.append(recs[0]["priority"])
        revenues_saved.append(recs[0]["estimated_revenue_saved"])

    df_eng["primary_recommendation"] = primary_recs
    df_eng["recommendation_priority"] = priorities
    df_eng["estimated_revenue_saved"] = revenues_saved

    # 8. Save Unified Customer Intelligence File
    logger.info("Saving: Exporting analytics files to disk...")
    # Keep key metrics
    export_cols = [
        "customer_id", "churn_probability", "predicted_ltv", "projected_future_ltv",
        "customer_segment", "rfm_persona", "intelligence_score", "intelligence_category",
        "primary_recommendation", "recommendation_priority", "estimated_revenue_saved"
    ]
    df_intelligence = df_eng[export_cols]
    df_intelligence.to_csv(REPORT_DIR / "customer_intelligence.csv", index=False)
    
    # Save a simplified segments map file
    df_segments = df_eng[["customer_id", "customer_segment"]]
    df_segments.to_csv(REPORT_DIR / "customer_segments.csv", index=False)

    # 9. Recommendation Analytics
    rec_counts = df_eng["primary_recommendation"].value_counts().to_frame("customers")
    rec_counts.to_csv(REPORT_DIR / "recommendation_statistics.csv")

    # 10. Generate Visual Analytics Charts
    sns.set_theme(style="whitegrid", palette="muted")

    # 10.1 LTV Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_eng, x="predicted_ltv", kde=True, bins=30, color="purple")
    plt.title("Customer Lifetime Value (LTV Historical Proxy) Distribution")
    plt.xlabel("LTV ($)")
    plt.savefig(PLOT_DIR / "ltv_distribution.png", dpi=150)
    plt.close()

    # 10.2 Customer Segments
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_eng, x="customer_segment", order=["Platinum", "Gold", "Silver", "Bronze"])
    plt.title("Customer Segments Distribution")
    plt.savefig(PLOT_DIR / "segment_distribution.png", dpi=150)
    plt.close()

    # 10.3 Cluster Visualisation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_eng, x="tenure_months", y="monthly_charges", hue="customer_segment", alpha=0.6)
    plt.title("Customer Segmentation Clusters (Tenure vs Monthly Charges)")
    plt.savefig(PLOT_DIR / "cluster_visualization.png", dpi=150)
    plt.close()

    # 10.4 RFM Heatmap
    plt.figure(figsize=(8, 6))
    rfm_matrix = df_rfm.groupby(["R_score", "F_score"])["M_score"].mean().unstack().fillna(0)
    sns.heatmap(rfm_matrix, annot=True, cmap="Purples", fmt=".2f")
    plt.title("RFM Heatmap: Mean Monetary Score by Recency & Frequency")
    plt.xlabel("Frequency Score")
    plt.ylabel("Recency Score")
    plt.savefig(PLOT_DIR / "rfm_heatmap.png", dpi=150)
    plt.close()

    # 10.5 Recommendation Chart
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_eng, y="primary_recommendation", hue="recommendation_priority")
    plt.title("Retention Recommendations Priority Distribution")
    plt.xlabel("Customer Count")
    plt.savefig(PLOT_DIR / "recommendation_chart.png", dpi=150)
    plt.close()

    # 10.6 Revenue Distribution
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_eng, x="customer_segment", y="total_charges", order=["Platinum", "Gold", "Silver", "Bronze"])
    plt.title("Accumulated Revenue Distribution by Customer Segment")
    plt.savefig(PLOT_DIR / "revenue_distribution.png", dpi=150)
    plt.close()

    # 10.7 Segment Comparison
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_eng, x="customer_segment", y="churn_probability", order=["Platinum", "Gold", "Silver", "Bronze"])
    plt.title("Churn Probability distribution by Customer Segment")
    plt.savefig(PLOT_DIR / "segment_comparison.png", dpi=150)
    plt.close()

    # 10.8 Score Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_eng, x="intelligence_score", kde=True, bins=30, color="green")
    plt.title("Customer Intelligence Score (0-100) Distribution")
    plt.xlabel("Intelligence Score")
    plt.savefig(PLOT_DIR / "score_distribution.png", dpi=150)
    plt.close()

    # 10.9 Top Revenue Customers
    plt.figure(figsize=(8, 4))
    top_rev = df_eng.sort_values(by="total_charges", ascending=False).head(10)
    sns.barplot(data=top_rev, x="total_charges", y="customer_id", color="gold")
    plt.title("Top 10 Customers by Accumulated Revenue")
    plt.xlabel("Revenue ($)")
    plt.savefig(PLOT_DIR / "top_revenue_customers.png", dpi=150)
    plt.close()

    # 10.10 Business Impact Dashboard Plot
    plt.figure(figsize=(8, 5))
    impact_data = df_eng.groupby("recommendation_priority")["estimated_revenue_saved"].sum().reset_index()
    sns.barplot(data=impact_data, x="recommendation_priority", y="estimated_revenue_saved", palette="viridis")
    plt.title("Estimated Potential Revenue Saved by recommendation Priority")
    plt.ylabel("Revenue Saved ($)")
    plt.savefig(PLOT_DIR / "business_dashboard.png", dpi=150)
    plt.close()

    # 11. Write Markdown Reports
    write_reports(df_eng, ltv_meta, stats_k_df)

    logger.info("Intelligence Pipeline: Completed successfully.")


def write_reports(df_eng: pd.DataFrame, ltv_meta: Dict[str, Any], stats_k_df: pd.DataFrame) -> None:
    """
    Generate the 6 required markdown intelligence reports.
    """
    # Helper statistics
    seg_counts = df_eng["customer_segment"].value_counts().to_dict()
    persona_counts = df_eng["rfm_persona"].value_counts().to_dict()
    total_rev = df_eng["total_charges"].sum()
    total_rev_saved = df_eng["estimated_revenue_saved"].sum()

    # 1. ltv_summary.md
    with open(REPORT_DIR / "ltv_summary.md", "w", encoding="utf-8") as f:
        f.write("# Customer Lifetime Value (LTV) Prediction Summary\n\n")
        f.write("This report documents the LTV prediction regression performance and future forecasts.\n\n")
        f.write("## 1. Regression Model Comparison (Historical Proxy)\n")
        f.write(f"- **Selected Best Model**: {ltv_meta['best_model']}\n")
        f.write(f"- **R² Coefficient of Determination**: {ltv_meta['r2']:.4f}\n")
        f.write(f"- **Root Mean Squared Error (RMSE)**: ${ltv_meta['rmse']:.2f}\n")
        f.write(f"- **Mean Absolute Error (MAE)**: ${ltv_meta['mae']:.2f}\n")
        f.write(f"- **Mean Absolute Percentage Error (MAPE)**: {ltv_meta['mape']:.2f}%\n\n")
        f.write("## 2. Hybrid LTV Target Definition\n")
        f.write("We use `total_charges` as a historical proxy of accumulated spend. We then estimate the **Projected Future LTV** via the expected remaining lifetime:\n")
        f.write("$$\\text{Remaining Lifetime (months)} = \\max\\left(0, \\frac{1}{\\text{Churn Probability}} - \\text{Tenure}\\right)$$\n")
        f.write("$$\\text{Projected Future LTV} = \\text{Remaining Lifetime} \\times \\text{Monthly Charges}$$\n\n")
        f.write(f"- **Total Projected Future LTV Potential**: ${df_eng['projected_future_ltv'].sum():,.2f}\n")
        f.write(f"- **Average Projected Future LTV**: ${df_eng['projected_future_ltv'].mean():.2f} per customer\n")

    # 2. segment_profiles.md
    with open(REPORT_DIR / "segment_profiles.md", "w", encoding="utf-8") as f:
        f.write("# Customer Segmentation Profiles Report\n\n")
        f.write("This report outlines the characteristics of the customer segment clusters generated via K-Means.\n\n")
        
        segments_list = ["Platinum", "Gold", "Silver", "Bronze"]
        for seg in segments_list:
            df_seg = df_eng[df_eng["customer_segment"] == seg]
            if len(df_seg) == 0:
                continue
            
            f.write(f"## 🏆 {seg} Customer Segment Profile\n")
            f.write(f"- **Customer Count**: {len(df_seg)} ({len(df_seg)/len(df_eng)*100:.1f}%)\n")
            f.write(f"- **Average Historical LTV**: ${df_seg['predicted_ltv'].mean():.2f}\n")
            f.write(f"- **Average Monthly Charges**: ${df_seg['monthly_charges'].mean():.2f}\n")
            f.write(f"- **Average Tenure**: {df_seg['tenure_months'].mean():.1f} months\n")
            f.write(f"- **Average Churn Risk**: {df_seg['churn_probability'].mean()*100:.1f}%\n")
            
            # Find top payment/contract methods
            top_contract = df_seg["contract_type"].mode().get(0, "Unknown")
            top_payment = df_seg["payment_method"].mode().get(0, "Unknown")
            f.write(f"- **Top Contract Type**: {top_contract}\n")
            f.write(f"- **Top Payment Method**: {top_payment}\n")
            
            campaign = "Standard Loyalty Campaign"
            if seg == "Platinum":
                campaign = "VIP Loyalty Rewards & High-Touch Service"
            elif seg == "Gold":
                campaign = "Paperless Auto-pay Incentives & Bundled Services"
            elif seg == "Silver":
                campaign = "Contract Extension Upgrade Campaigns"
            elif seg == "Bronze":
                campaign = "Aggressive price sensitivity promotions & basic bundles"
                
            f.write(f"- **Recommended Marketing Campaign**: {campaign}\n\n")

    # 3. executive_summary.md
    with open(REPORT_DIR / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write("# Executive Customer Intelligence Summary\n\n")
        f.write(f"- **Date of Report**: {datetime.utcnow().date().isoformat()}\n")
        f.write(f"- **Total Active Customers Analyzed**: {len(df_eng)}\n")
        f.write(f"- **Overall Database Churn Rate**: {df_eng['churn'].mean()*100:.1f}%\n")
        f.write(f"- **Overall Projected Churn Risk (Probabilistic)**: {df_eng['churn_probability'].mean()*100:.1f}%\n\n")
        
        f.write("## 1. Segment Customer Share\n")
        for k, v in seg_counts.items():
            f.write(f"- **{k}**: {v} customers ({v/len(df_eng)*100:.1f}%)\n")
            
        f.write("\n## 2. RFM Personas Share\n")
        for k, v in list(persona_counts.items())[:5]:
            f.write(f"- **{k}**: {v} customers ({v/len(df_eng)*100:.1f}%)\n")
            
        f.write("\n## 3. High-Risk Retention Actionables\n")
        critical_count = len(df_eng[df_eng["recommendation_priority"] == "Critical"])
        high_count = len(df_eng[df_eng["recommendation_priority"] == "High"])
        f.write(f"- **Critical Priority Actions Required**: {critical_count} customers\n")
        f.write(f"- **High Priority Actions Required**: {high_count} customers\n")

    # 4. business_impact.md
    with open(REPORT_DIR / "business_impact.md", "w", encoding="utf-8") as f:
        f.write("# Business Impact & Financial Simulation Report\n\n")
        f.write("We estimate the financial savings from acting on model recommendation triggers.\n\n")
        
        f.write(f"- **Total Revenue at Attrition Risk (Probabilistic)**: ${df_eng['churn_probability'].dot(df_eng['total_charges']):,.2f}\n")
        f.write(f"- **Maximum Potential Revenue Saved**: ${total_rev_saved:,.2f}\n\n")
        
        f.write("## 1. Financial Impact Simulation (50% Response Assumption)\n")
        f.write("If **50%** of targeted high-risk customers accept our proactive retention/upgrade recommendations:\n")
        f.write(f"- **Targeted Customers**: {len(df_eng[df_eng['churn_probability'] >= 0.40])} accounts\n")
        f.write(f"- **Estimated Financial Savings**: **${total_rev_saved * 0.50:,.2f}**\n")
        f.write(f"- **Targeted Spend ROI**: Proactive offers (discounts/setup billing credit) typically cost $15-$30 per account, yielding an expected 5-10x return on retained contract margins.\n")

    # 5. recommendation_summary.md
    with open(REPORT_DIR / "recommendation_summary.md", "w", encoding="utf-8") as f:
        f.write("# Recommendation Engine Output Summary\n\n")
        f.write("This report summaries the volumes and priority of all triggered customer loyalty campaigns.\n\n")
        
        # Group counts
        recs_summary = df_eng.groupby(["primary_recommendation", "recommendation_priority"])["customer_id"].count().reset_index()
        f.write("| Campaign Recommendation | Priority | Targeted Customers | Expected Saved Rev |\n")
        f.write("| --- | --- | --- | --- |\n")
        for _, row in recs_summary.iterrows():
            sub_df = df_eng[df_eng["primary_recommendation"] == row["primary_recommendation"]]
            rev_sum = sub_df["estimated_revenue_saved"].sum()
            f.write(f"| {row['primary_recommendation']} | {row['recommendation_priority']} | {row['customer_id']} | ${rev_sum:,.2f} |\n")

    # 6. customer_intelligence.md (Unified detailed file)
    with open(REPORT_DIR / "customer_intelligence.md", "w", encoding="utf-8") as f:
        f.write("# Unified Customer Intelligence scoring Methodology\n\n")
        f.write("The Unified **Customer Intelligence Score (0-100)** is computed as a weighted index of value and risk metrics:\n\n")
        
        f.write("## 1. Scoring Formula\n")
        f.write("```text\n")
        f.write("Score = 30% * (1.0 - Churn Probability) * 100\n")
        f.write("      + 30% * (log(Predicted LTV) / log(Max LTV)) * 100\n")
        f.write("      + 20% * (Tenure / 72 months) * 100\n")
        f.write("      + 20% * (Services / 8 services) * 100\n")
        f.write("```\n\n")
        
        f.write("## 2. Category Share Breakdown\n")
        score_cats = df_eng["intelligence_category"].value_counts().to_dict()
        for k, v in score_cats.items():
            f.write(f"- **{k}** (Count: {v}): average score of {df_eng[df_eng['intelligence_category'] == k]['intelligence_score'].mean():.1f}\n")


if __name__ == "__main__":
    main()
