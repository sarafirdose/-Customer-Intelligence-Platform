"""
Recommendation Center Page - Streamlit Dashboard.

Group retention actions by priority and estimate campaign financial savings.
"""

import streamlit as st
import pandas as pd

from dashboard.components.cards import render_executive_header, render_kpi_card, render_ai_copilot_widget
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.tables import render_interactive_table
from dashboard.components.charts import plot_recommendation_chart
from dashboard.utils.cache import load_global_intelligence_data


def render_recommendations_page():
    """
    Renders recommendations analytics.
    """
    render_executive_header(
        title="🎯 Proactive Retention Recommendation Center",
        subtitle="Explore rules-based upselling promotions, check priority tiers, and analyze expected financial retention savings.",
        badge_text="AI Action Engine v1.0"
    )

    df_intel = load_global_intelligence_data()

    if df_intel.empty:
        st.error("No analytics data loaded.")
        return

    # Apply global sidebar filters
    df_filtered = render_sidebar_filters(df_intel)

    # Financial ROI KPI strip
    total_savings = df_filtered["estimated_revenue_saved"].sum() if not df_filtered.empty else 0.0
    crit_count = len(df_filtered[df_filtered["recommendation_priority"] == "Critical"])
    high_count = len(df_filtered[df_filtered["recommendation_priority"] == "High"])
    med_count = len(df_filtered[df_filtered["recommendation_priority"] == "Medium"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(f"${total_savings:,.2f}", "Total Campaign ROI Savings", border_color="#10B981", trend="Net Recoverable", trend_type="positive")
    with c2:
        render_kpi_card(f"{crit_count:,}", "Critical Actions", border_color="#EF4444", trend="Immediate Outreach", trend_type="negative")
    with c3:
        render_kpi_card(f"{high_count:,}", "High Priority Actions", border_color="#F59E0B", trend="Targeted Offer", trend_type="neutral")
    with c4:
        render_kpi_card(f"{med_count:,}", "Medium Priority Actions", border_color="#3B82F6", trend="Standard Plan", trend_type="positive")

    st.divider()

    # Plot recommendation volume split
    st.plotly_chart(plot_recommendation_chart(df_filtered), use_container_width=True)

    st.divider()

    # Campaign list explorer
    st.subheader("📋 Campaign Targets Watchlist")
    
    unique_campaigns = ["All"] + sorted(list(df_filtered["primary_recommendation"].dropna().unique()))
    selected_camp = st.selectbox("Filter Watchlist by Retention Campaign Type", unique_campaigns)

    df_watchlist = df_filtered.copy()
    if selected_camp != "All":
        df_watchlist = df_watchlist[df_watchlist["primary_recommendation"] == selected_camp]

    display_cols = [
        "customer_id", "primary_recommendation", "recommendation_priority",
        "churn_probability", "predicted_ltv", "estimated_revenue_saved"
    ]
    render_interactive_table(df_watchlist[display_cols].sort_values(by="estimated_revenue_saved", ascending=False), page_size=10, key="camp_table")


if __name__ == "__main__":
    render_recommendations_page()
