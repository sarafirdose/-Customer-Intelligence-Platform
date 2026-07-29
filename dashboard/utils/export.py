"""
Export utilities for the Streamlit dashboard.

Allows downloading analytics tables as Excel or CSV and Plotly charts as images.
"""

import io
import pandas as pd
import streamlit as st


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame to a CSV byte stream.
    """
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame to an Excel byte stream.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


def download_button_dataframe(df: pd.DataFrame, file_name: str, button_text: str = "Download Data"):
    """
    Generate downloadable download buttons for DataFrame exports.
    """
    csv_bytes = convert_df_to_csv(df)
    st.download_button(
        label=button_text,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        key=f"dl_{file_name}"
    )
