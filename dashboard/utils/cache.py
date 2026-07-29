"""
Caching utilities for the Streamlit dashboard.

Implements cached file loading to accelerate dashboard performance.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"


@st.cache_data(ttl=600)
def load_global_intelligence_data() -> pd.DataFrame:
    """
    Load pre-compiled customer intelligence dataset from reports/.
    """
    file_path = REPORT_DIR / "customer_intelligence.csv"
    if not file_path.exists():
        # Fallback empty dataframe with schemas
        return pd.DataFrame(columns=[
            "customer_id", "churn_probability", "predicted_ltv", "projected_future_ltv",
            "customer_segment", "rfm_persona", "intelligence_score", "intelligence_category",
            "primary_recommendation", "recommendation_priority", "estimated_revenue_saved"
        ])
    return pd.read_csv(file_path)


@st.cache_data(ttl=600)
def load_rfm_analysis_data() -> pd.DataFrame:
    """
    Load pre-compiled RFM scores and categories.
    """
    file_path = REPORT_DIR / "rfm_analysis.csv"
    if not file_path.exists():
        return pd.DataFrame(columns=["customer_id", "R_score", "F_score", "M_score", "rfm_score", "persona"])
    return pd.read_csv(file_path)


@st.cache_data(ttl=600)
def load_report_markdown(report_name: str) -> str:
    """
    Load report markdown file content.
    """
    file_path = REPORT_DIR / report_name
    if not file_path.exists():
        return f"Report {report_name} is not available."
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
