"""
Enterprise Sidebar Navigation & Global Filters Component.

Renders modern sidebar with workspace switcher, command search trigger, and cohort filters.
"""

from typing import Tuple
import pandas as pd
import streamlit as st


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render enterprise sidebar navigation, workspace selector, and dynamic filters.
    """
    # 1. Workspace Selector
    st.sidebar.markdown(
        """
        <div class="sidebar-workspace">
            <div style="font-size: 1.4rem;">🌐</div>
            <div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC;">Jio Telecom Enterprise</div>
                <div style="font-size: 0.72rem; color: #34D399; font-weight: 600;">Prod Cluster • v2.4</div>
            </div>
        </div>
        <div class="sidebar-cmd-k">
            <span>🔍 Command Search</span>
            <kbd>Ctrl + K</kbd>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🎛️ Cohort Filters")

    # 2. Segment filter
    segments = ["All"] + sorted(list(df["customer_segment"].dropna().unique()))
    selected_segment = st.sidebar.selectbox("Customer Segment", segments, key="filter_seg")

    # 3. Churn Risk category
    risk_options = ["All", "High Risk (>= 0.61)", "Medium Risk (0.40 - 0.61)", "Low Risk (< 0.40)"]
    selected_risk = st.sidebar.selectbox("Attrition Risk Class", risk_options, key="filter_risk")

    # 4. Score Category
    score_cats = ["All"] + sorted(list(df["intelligence_category"].dropna().unique()))
    selected_score_cat = st.sidebar.selectbox("Intelligence Class", score_cats, key="filter_score")

    # 5. RFM Persona
    personas = ["All"] + sorted(list(df["rfm_persona"].dropna().unique()))
    selected_persona = st.sidebar.selectbox("RFM Persona", personas, key="filter_rfm")

    # Apply filters dynamically
    df_filtered = df.copy()

    if selected_segment != "All":
        df_filtered = df_filtered[df_filtered["customer_segment"] == selected_segment]

    if selected_risk == "High Risk (>= 0.61)":
        df_filtered = df_filtered[df_filtered["churn_probability"] >= 0.61]
    elif selected_risk == "Medium Risk (0.40 - 0.61)":
        df_filtered = df_filtered[(df_filtered["churn_probability"] >= 0.40) & (df_filtered["churn_probability"] < 0.61)]
    elif selected_risk == "Low Risk (< 0.40)":
        df_filtered = df_filtered[df_filtered["churn_probability"] < 0.40]

    if selected_score_cat != "All":
        df_filtered = df_filtered[df_filtered["intelligence_category"] == selected_score_cat]

    if selected_persona != "All":
        df_filtered = df_filtered[df_filtered["rfm_persona"] == selected_persona]

    st.sidebar.divider()
    
    # Dataset Filter Counter Badge
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; color: #A5B4FC; text-align: center; margin-bottom: 16px;">
            📊 Active Cohort: <b>{len(df_filtered):,}</b> / <b>{len(df):,}</b> Subscribers
        </div>
        """,
        unsafe_allow_html=True,
    )

    # User Profile Footer
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #6366F1, #8B5CF6); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: white;">
                VP
            </div>
            <div>
                <div style="font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">Executive Officer</div>
                <div style="font-size: 0.7rem; color: #64748B;">Chief Analytics Team</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return df_filtered
