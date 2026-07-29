"""
LTV Analytics Page - Streamlit Dashboard.

Visualizes historical spends, projected customer lifetimes, and billing distributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.components.cards import render_executive_header, render_kpi_card, render_ai_copilot_widget
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.charts import plot_ltv_distribution, plot_top_revenue_bar
from dashboard.utils.cache import load_global_intelligence_data


def render_ltv_page():
    """
    Renders LTV analytics reports.
    """
    render_executive_header(
        title="💰 Customer Lifetime Value (LTV) Analytics",
        subtitle="Forecast future customer valuations, inspect contract billing speeds, and evaluate historical revenue accumulations.",
        badge_text="LTV Regression Engine"
    )

    df_intel = load_global_intelligence_data()

    if df_intel.empty:
        st.error("No analytics data loaded.")
        return

    # Apply global sidebar filters
    df_filtered = render_sidebar_filters(df_intel)

    # Metrics overview
    total_projected_revenue = df_filtered["projected_future_ltv"].sum() if not df_filtered.empty else 0.0
    avg_projected_ltv = df_filtered["projected_future_ltv"].mean() if not df_filtered.empty else 0.0
    avg_hist_ltv = df_filtered["predicted_ltv"].mean() if not df_filtered.empty else 0.0

    st.markdown("### 📈 LTV Forecast & Revenue Portfolio")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        render_kpi_card(f"${total_projected_revenue:,.2f}", "Total Projected LTV", border_color="#6366F1", trend="+8.4% YoY", trend_type="positive")
    with f_col2:
        render_kpi_card(f"${avg_projected_ltv:,.2f}", "Avg Projected LTV", border_color="#8B5CF6", trend="Future horizon", trend_type="positive")
    with f_col3:
        render_kpi_card(f"${avg_hist_ltv:,.2f}", "Avg Historical LTV", border_color="#3B82F6", trend="Realized spend", trend_type="positive")
    with f_col4:
        ratio = (total_projected_revenue / (df_filtered["predicted_ltv"].sum() + 1e-6)) if not df_filtered.empty else 1.0
        render_kpi_card(f"{ratio:.2f}x", "LTV Expansion Multiple", border_color="#10B981", trend="Growth factor", trend_type="positive")

    st.divider()

    # Visualizations
    col_dist, col_top = st.columns(2)
    with col_dist:
        st.plotly_chart(plot_ltv_distribution(df_filtered), use_container_width=True)
    with col_top:
        st.plotly_chart(plot_top_revenue_bar(df_filtered), use_container_width=True)


if __name__ == "__main__":
    render_ltv_page()
