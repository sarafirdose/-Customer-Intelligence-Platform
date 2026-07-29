"""
Data Tables Component for Streamlit Dashboard.

Exposes helpers to render customized interactive tables.
"""

import streamlit as st
import pandas as pd


def render_interactive_table(df: pd.DataFrame, page_size: int = 10, key: str = "table"):
    """
    Render a paginated, search-aligned Streamlit dataframe.
    """
    if df.empty:
        st.warning("No records to display.")
        return

    # Pagination controls
    total_rows = len(df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)
    
    col1, col2 = st.columns([8, 2])
    with col2:
        current_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{key}_page"
        )
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Display subset
    df_subset = df.iloc[start_idx:end_idx]
    
    st.dataframe(df_subset, use_container_width=True)
    st.caption(f"Showing {start_idx + 1} - {end_idx} of {total_rows} accounts")
