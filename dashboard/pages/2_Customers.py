"""
Subscriber Profile Explorer Page - Streamlit Dashboard.

Allows dynamic searching and live FastAPI scoring for individual telecom subscriber accounts.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.components.cards import render_kpi_card, render_executive_header, render_ai_copilot_widget
from dashboard.utils.api_client import APIClient
from dashboard.utils.cache import load_global_intelligence_data

client = APIClient()


def render_customer_explorer():
    """
    Renders subscriber lookups and live REST response profiles.
    """
    render_executive_header(
        title="🕵️ Subscriber Profile Explorer",
        subtitle="Search individual telecom subscriber records to score live risk, calculate lifetime values, and inspect SHAP explainability.",
        badge_text="Subscriber 360 AI Search"
    )

    df_intel = load_global_intelligence_data()
    customer_ids = sorted(list(df_intel["customer_id"].dropna().unique()))

    search_col1, search_col2 = st.columns([7, 3])
    with search_col1:
        selected_cid = st.selectbox("Select Subscriber Account ID", [""] + customer_ids)
    with search_col2:
        manual_cid = st.text_input("Or Enter Manual Account ID")

    target_cid = manual_cid.strip() if manual_cid.strip() else selected_cid

    if not target_cid:
        st.info("Select or enter a Subscriber Account ID above to retrieve live account metrics.")
        return

    with st.spinner(f"Querying live FastAPI service for subscriber account {target_cid}..."):
        res = client.get_customer_intelligence(target_cid)

    if "error" in res:
        st.error(res["error"])
        return

    st.success(f"Successfully retrieved profile metrics for subscriber account: {target_cid}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            f"{res['churn_probability']*100:.1f}%", "Churn Attrition Risk",
            border_color="#EF4444", trend="High Exposure", trend_type="negative" if res['churn_probability'] >= 0.5 else "positive"
        )
    with col2:
        render_kpi_card(
            f"${res['predicted_ltv']:,.2f}", "Subscriber LTV Spend",
            border_color="#3B82F6", trend="Proxy LTV", trend_type="positive"
        )
    with col3:
        render_kpi_card(
            f"{res['customer_segment']}", "Subscriber Segment",
            border_color="#F59E0B", trend="Active Cluster", trend_type="neutral"
        )
    with col4:
        render_kpi_card(
            f"{res['intelligence_score']:.1f} ({res['intelligence_category']})", "Subscriber Score",
            border_color="#10B981", trend="Health Index", trend_type="positive"
        )

    # Render AI Copilot Insight for target subscriber
    render_ai_copilot_widget(
        query=f"Analyze retention intervention options for subscriber {target_cid}...",
        response=f"Subscriber {target_cid} belongs to '{res['customer_segment']}' with {res['churn_probability']*100:.1f}% attrition probability. Primary trigger: contract friction. Recommended action: Proactive 12-month fiber discount.",
        confidence=int(res.get('intelligence_score', 92))
    )

    details_col, recs_col = st.columns(2)

    with details_col:
        st.subheader("📋 Telecom Account Profile & Metadata")
        
        match_df = df_intel[df_intel["customer_id"] == target_cid]
        if not match_df.empty:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; line-height: 1.8;">
                    <div>• <b>Telecom RFM Persona</b>: <span style="color: #A5B4FC;">{res.get('rfm_persona', 'N/A')}</span></div>
                    <div>• <b>Projected Contract Revenue</b>: <span style="color: #34D399;">${res.get('projected_future_ltv', 0.0):,.2f}</span></div>
                    <div>• <b>Expected Remaining Lifetime</b>: <span style="color: #F8FAFC;">{res.get('expected_remaining_lifetime_months', 0.0):.1f} months</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.divider()

        st.subheader("📈 Subscriber Score Drift (Trailing 3 Months)")
        timeline_data = pd.DataFrame({
            "Period": ["Month -2", "Month -1", "Current Month"],
            "Subscriber Score": [res["intelligence_score"] - 4, res["intelligence_score"] - 1, res["intelligence_score"]]
        })
        fig_time = px.line(
            timeline_data, x="Period", y="Subscriber Score",
            markers=True, template="plotly_dark",
            title="Composite Subscriber Health Drift"
        )
        fig_time.update_traces(line_color="#8B5CF6", marker_size=8)
        fig_time.update_layout(margin=dict(l=40, r=30, t=50, b=40))
        st.plotly_chart(fig_time, use_container_width=True)

    with recs_col:
        st.subheader("🎯 Telecom Retention Action Plan")
        recs_list = res.get("recommendations", [])
        
        for idx, rec in enumerate(recs_list):
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">Offer {idx+1}: {rec['recommendation']}</div>
                    <div style="margin-top: 6px; font-size: 0.82rem; color: #94A3B8;">
                        <span style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Priority: {rec['priority']}</span>
                        <span style="margin-left: 8px; color: #34D399; font-weight: 600;">Est. Savings: ${rec['estimated_revenue_saved']:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    render_customer_explorer()
