"""
Churn Analytics Page - Streamlit Dashboard.

Visualizes classifier ROC/PR metrics, confusion matrices, and the high-risk account watchlist.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from dashboard.components.cards import render_executive_header, render_kpi_card, render_ai_copilot_widget
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.tables import render_interactive_table
from dashboard.utils.cache import load_global_intelligence_data

BASE_DIR = Path(__file__).resolve().parents[2]
PLOT_DIR = BASE_DIR / "reports" / "plots"


def render_churn_page():
    """
    Renders churn model metrics.
    """
    render_executive_header(
        title="📉 Churn Risk Analytics & Watchlist",
        subtitle="Examine classifier evaluation curves, model calibrations, and inspect the high-risk customer outreach watchlist.",
        badge_text="LGBM Classifier Engine v1.0"
    )

    df_intel = load_global_intelligence_data()

    if df_intel.empty:
        st.error("No analytics data loaded.")
        return

    # Apply global sidebar filters
    df_filtered = render_sidebar_filters(df_intel)

    high_risk_df = df_filtered[df_filtered["churn_probability"] >= 0.61].sort_values(
        by="churn_probability", ascending=False
    )
    med_risk_df = df_filtered[(df_filtered["churn_probability"] >= 0.40) & (df_filtered["churn_probability"] < 0.61)]

    # Metrics strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(f"{len(high_risk_df):,}", "High Risk Watchlist", border_color="#EF4444", trend="≥ 0.61 Threshold", trend_type="negative")
    with c2:
        render_kpi_card(f"{len(med_risk_df):,}", "Medium Risk Watchlist", border_color="#F59E0B", trend="0.40–0.61 Range", trend_type="neutral")
    with c3:
        avg_risk_score = df_filtered["churn_probability"].mean() * 100.0 if not df_filtered.empty else 0.0
        render_kpi_card(f"{avg_risk_score:.1f}%", "Cohort Mean Risk", border_color="#3B82F6", trend="Baseline 31.8%", trend_type="positive")
    with c4:
        render_kpi_card("0.847", "Model ROC-AUC", border_color="#10B981", trend="Production Grade", trend_type="positive")

    st.divider()

    # Watchlist header
    st.subheader("🚨 High-Risk Customer Outreach Watchlist")
    
    # Select key columns for the watchlist
    watchlist_cols = [
        "customer_id", "churn_probability", "customer_segment", "rfm_persona",
        "intelligence_score", "primary_recommendation", "recommendation_priority"
    ]
    render_interactive_table(high_risk_df[watchlist_cols], page_size=10, key="watchlist")

    st.divider()

    # Model Evaluation curves block
    st.subheader("📊 Classifier Performance & Diagnostic Curves")
    col_roc, col_pr = st.columns(2)
    with col_roc:
        roc_path = PLOT_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curve")
        else:
            st.warning("ROC Curve image not found. Run scripts/train_all.py first.")
            
    with col_pr:
        pr_path = PLOT_DIR / "pr_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve")
        else:
            st.warning("PR Curve image not found.")

    st.divider()

    col_cal, col_cm = st.columns(2)
    with col_cal:
        cal_path = PLOT_DIR / "calibration_curve.png"
        if cal_path.exists():
            st.image(str(cal_path), caption="Probability Calibration Curve")
        else:
            st.warning("Calibration Curve image not found.")
            
    with col_cm:
        cm_path = PLOT_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix at 0.61 Threshold")
        else:
            st.warning("Confusion Matrix image not found.")


if __name__ == "__main__":
    render_churn_page()
