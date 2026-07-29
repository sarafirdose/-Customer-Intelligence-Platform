"""
Batch Analysis Page - Streamlit Dashboard.

Allows uploading customer ID CSVs, scoring them via FastAPI, and exporting Excel/CSV tables.
"""

import streamlit as st
import pandas as pd

from dashboard.components.cards import render_executive_header, render_kpi_card
from dashboard.utils.api_client import APIClient
from dashboard.utils.export import convert_df_to_csv, convert_df_to_excel

# Initialize API Client
client = APIClient()


def render_batch_page():
    """
    Renders batch CSV analysis widgets.
    """
    render_executive_header(
        title="📥 Batch Analysis Center",
        subtitle="Upload lists of customer account IDs to score risks, calculate LTV forecasts, and generate campaigns in bulk.",
        badge_text="Async Batch Pipeline v1.0"
    )

    st.markdown("### 1. Upload Customer ID File")

    # Sample CSV helper
    sample_csv_content = (
        "customer_id\n"
        "7590-VHVEG\n"
        "5575-GNVDE\n"
        "3668-QPYBK\n"
        "7795-CFOCW\n"
        "9237-HQJOC\n"
        "9305-CDSKC\n"
        "1452-KTVCR\n"
        "6713-OKOMC\n"
        "7892-POOKP\n"
        "6388-TABGU\n"
    )
    st.download_button(
        label="📄 Download Sample Subscriber ID CSV File",
        data=sample_csv_content,
        file_name="sample_subscriber_ids.csv",
        mime="text/csv",
        help="Click to download a ready-to-use CSV template containing valid Subscriber IDs."
    )

    uploaded_file = st.file_uploader("Choose a CSV file containing customer IDs", type=["csv"])

    if not uploaded_file:
        st.info("Please upload a CSV file. The file must contain a column for customer IDs (e.g. named 'customer_id' or 'customerID').")
        return

    # Read CSV
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded File Preview:")
        st.dataframe(df_uploaded.head(5))
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return

    # Find customer ID column
    id_col = None
    for col in df_uploaded.columns:
        if str(col).strip().lower() in ["customer_id", "customerid", "id", "account_id"]:
            id_col = col
            break

    if not id_col:
        st.error("Could not find a valid Customer ID column in the uploaded CSV. Ensure it has a column header named 'customer_id'.")
        return

    customer_ids = [str(x).strip() for x in df_uploaded[id_col].dropna().unique()]
    st.info(f"Identified {len(customer_ids)} unique customer IDs for batch evaluation.")

    # Trigger batch scoring
    if st.button("🚀 Execute Batch Scoring Pipeline"):
        with st.spinner("Processing batch scoring via FastAPI..."):
            res_list = client.batch_score_customers(customer_ids)

        if not res_list or "error" in res_list[0]:
            st.error(res_list[0].get("error", "Batch scoring failed."))
            return

        # Convert result list of dicts to DataFrame
        df_results = pd.DataFrame(res_list)
        
        st.success("Batch scoring completed successfully!")
        
        # Display predictions summary
        st.subheader("📊 Output Preview")
        st.dataframe(df_results.head(10))

        # Download buttons
        st.subheader("💾 Export Analytics Results")
        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            csv_data = convert_df_to_csv(df_results)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name="batch_scored_intelligence.csv",
                mime="text/csv"
            )
        with col_xlsx:
            excel_data = convert_df_to_excel(df_results)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name="batch_scored_intelligence.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    render_batch_page()
