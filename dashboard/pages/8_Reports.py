"""
Reports Page - Streamlit Dashboard.

Allows viewing and downloading executive summaries, business impacts, LTV metrics, and segmentation files.
"""

from pathlib import Path
import streamlit as st

from dashboard.components.cards import render_executive_header
from dashboard.utils.cache import load_report_markdown

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"


def render_reports_page():
    """
    Renders reports selection and markdown viewer.
    """
    render_executive_header(
        title="📂 Reports & Documentation Center",
        subtitle="Read and download executive summaries, financial impact simulations, and classifier evaluations.",
        badge_text="Executive Knowledge Base"
    )

    reports_map = {
        "Executive Summary": "executive_summary.md",
        "Business Impact & ROI Simulation": "business_impact.md",
        "LTV Modeling Performance": "ltv_summary.md",
        "K-Means Segment Profiles": "segment_profiles.md",
        "Campaign Recommendations Volume": "recommendation_summary.md",
        "Unified Scoring Methodology": "customer_intelligence.md"
    }

    # Select report
    selected_rep_title = st.selectbox("Choose Report to View", list(reports_map.keys()))
    selected_file = reports_map[selected_rep_title]

    # Load and display report
    report_content = load_report_markdown(selected_file)

    st.divider()

    # Markdown preview inside glassmorphic card container
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 24px; margin-bottom: 20px;">
            {report_content}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Download button for raw markdown file
    st.download_button(
        label=f"💾 Download {selected_rep_title} (.md)",
        data=report_content,
        file_name=selected_file,
        mime="text/markdown"
    )


if __name__ == "__main__":
    render_reports_page()
