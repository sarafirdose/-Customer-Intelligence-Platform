"""
Executive Subscriber Summary Page - Single-Page Telecom High-Level Overview.

Designed for executive leadership and telecom stakeholders:
  - High-level KPI metrics (Total Subscribers, High Risk, Revenue at Risk, Expected Savings)
  - Subscriber Intelligence Score & Model Accuracy highlights
  - Telecom Strategic Retention Recommendations & ROI Summary
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_executive_header, render_kpi_card

def render_executive_summary_page() -> None:
    render_executive_header(
        title="👑 Executive Subscriber Summary — Telecom Intelligence Platform",
        subtitle="One-page strategic overview of telecom subscriber health, churn exposure, revenue risk, and AI model performance.",
        badge_text="Executive Briefing v2.4"
    )

    # -------------------------------------------------------------------
    # 1. Executive Top KPI Grid
    # -------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("7,043", "Total Subscribers", border_color="#6366F1", trend="+3.8% YoY", trend_type="positive")
    with col2:
        render_kpi_card("2,255 (32.0%)", "High Risk Exposure", border_color="#EF4444", trend="≥ 0.61 Threshold", trend_type="negative")
    with col3:
        render_kpi_card("$1.82M", "Annual Revenue at Risk", border_color="#F59E0B", trend="Unmitigated Risk", trend_type="negative")
    with col4:
        render_kpi_card("$1.87M", "Expected ROI Savings", border_color="#10B981", trend="Net Recoverable", trend_type="positive")

    st.divider()

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        render_kpi_card("68 / 100", "Avg Subscriber Score", border_color="#06B6D4", trend="Health Index", trend_type="positive")
    with col6:
        render_kpi_card("84.7%", "Model Accuracy (ROC-AUC)", border_color="#8B5CF6", trend="LGBM Classifier", trend_type="positive")
    with col7:
        render_kpi_card("High-Value", "Top Subscriber Segment", border_color="#EC4899", trend="Core Revenue Driver", trend_type="positive")
    with col8:
        render_kpi_card("Month-to-Month", "Primary Churn Factor", border_color="#F59E0B", trend="Contract Friction", trend_type="neutral")

    st.divider()

    # -------------------------------------------------------------------
    # 2. Revenue Risk & Segment Opportunity Breakdown
    # -------------------------------------------------------------------
    st.subheader("📊 Strategic Revenue Exposure & Telecom Segment Health")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Revenue at Risk by Telecom Subscriber Segment")
        segment_data = pd.DataFrame([
            {"Segment": "High-Value Subscribers", "Subscribers": 1410, "Revenue_at_Risk": 620000, "Avg_Score": 88},
            {"Segment": "Loyal Subscribers", "Subscribers": 2150, "Revenue_at_Risk": 510000, "Avg_Score": 74},
            {"Segment": "Growth Subscribers", "Subscribers": 2230, "Revenue_at_Risk": 440000, "Avg_Score": 61},
            {"Segment": "Budget Subscribers", "Subscribers": 1253, "Revenue_at_Risk": 250000, "Avg_Score": 48},
        ])
        fig1 = px.bar(
            segment_data,
            x="Segment",
            y="Revenue_at_Risk",
            color="Segment",
            title="Revenue Exposure by Telecom Segment ($)",
            color_discrete_sequence=["#10B981", "#F59E0B", "#6366F1", "#EF4444"],
            template="plotly_dark",
        )
        fig1.update_layout(
            margin=dict(l=40, r=30, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### High-Risk Subscriber Exposure by Contract Type")
        contract_data = pd.DataFrame([
            {"Contract": "Month-to-Month", "High_Risk_Subscribers": 1655, "Percentage": 73.4},
            {"Contract": "One Year", "High_Risk_Subscribers": 380, "Percentage": 16.8},
            {"Contract": "Two Year", "High_Risk_Subscribers": 220, "Percentage": 9.8},
        ])
        fig2 = px.pie(
            contract_data,
            names="Contract",
            values="High_Risk_Subscribers",
            title="High-Risk Exposure by Contract Type",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark",
            hole=0.4,
        )
        fig2.update_layout(
            margin=dict(l=40, r=30, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    render_executive_summary_page()
