"""
Telecom Subscriber Segment Analytics Page - Streamlit Dashboard.

Visualizes K-Means clustering distributions, average segment spends, and risk comparison across High-Value, Loyal, Growth, and Budget subscriber tiers.
"""

import streamlit as st
import pandas as pd

from dashboard.components.cards import render_executive_header, render_kpi_card, render_ai_copilot_widget
from dashboard.components.filters import render_sidebar_filters
from dashboard.components.charts import (
    plot_segment_distribution,
    plot_cluster_scatter,
    plot_revenue_distribution_box,
    plot_segment_comparison_box
)
from dashboard.utils.cache import load_global_intelligence_data


def render_segments_page():
    """
    Renders the subscriber segments analytics layout.
    """
    render_executive_header(
        title="🏆 Telecom Subscriber Segments",
        subtitle="Evaluate subscriber cohort distributions, cluster boundaries, and risk comparison across High-Value, Loyal, Growth, and Budget subscriber tiers.",
        badge_text="K-Means Clustering Engine"
    )

    df_intel = load_global_intelligence_data()

    if df_intel.empty:
        st.error("No analytics data loaded.")
        return

    df_filtered = render_sidebar_filters(df_intel)

    # Segment Summary KPI strip
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card("4 Clusters", "Segmentation Tiers", border_color="#6366F1", subtext="K-Means Model v1.0")
    with s2:
        high_val_cnt = len(df_filtered[df_filtered["customer_segment"].str.contains("High-Value", case=False, na=False)])
        render_kpi_card(f"{high_val_cnt:,}", "High-Value Segment", border_color="#10B981", trend="Core Revenue Driver")
    with s3:
        loyal_cnt = len(df_filtered[df_filtered["customer_segment"].str.contains("Loyal", case=False, na=False)])
        render_kpi_card(f"{loyal_cnt:,}", "Loyal Subscribers", border_color="#3B82F6", trend="Low Attrition")
    with s4:
        budget_cnt = len(df_filtered[df_filtered["customer_segment"].str.contains("Budget", case=False, na=False)])
        render_kpi_card(f"{budget_cnt:,}", "Budget Segment", border_color="#F59E0B", trend="Price Sensitive")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_segment_distribution(df_filtered), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_cluster_scatter(df_filtered), use_container_width=True)

    st.divider()

    col_spend, col_churn = st.columns(2)
    with col_spend:
        st.plotly_chart(plot_revenue_distribution_box(df_filtered), use_container_width=True)
    with col_churn:
        st.plotly_chart(plot_segment_comparison_box(df_filtered), use_container_width=True)


if __name__ == "__main__":
    render_segments_page()
