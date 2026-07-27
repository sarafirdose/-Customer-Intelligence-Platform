"""
Customer Intelligence Platform Analytics Dashboard.

Streamlit dashboard showing predictions, churn distributions, customer segmentations,
and explainable feature attributions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page layout
st.set_page_config(
    page_title="Customer Intelligence Platform - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.2rem;
            color: #4B5563;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #F3F4F6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid #3B82F6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Customer Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">AI-Powered Customer Churn Prediction & Lifetime Value (LTV) Engine</div>',
    unsafe_allow_html=True,
)

# Sidebar configurations
st.sidebar.header("Navigation & Configurations")
page = st.sidebar.selectbox(
    "Select View",
    ["Overview Analytics", "Predict Customer Churn", "Model Performance & SHAP"],
)

if page == "Overview Analytics":
    st.header("Overview Analytics")

    # Metrics section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers Monitored", "1,250", "+12% MoM")
    with col2:
        st.metric("Average Churn Risk", "24.5%", "-2.1% MoM")
    with col3:
        st.metric("Estimated Total LTV", "$1.88M", "+8.4% MoM")
    with col4:
        st.metric("Retention ROI", "324%", "+15%")

    st.markdown("---")

    # Plotly mock charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Customer Tenure vs. Monthly Charges")
        mock_data = pd.DataFrame(
            {
                "Tenure (Months)": [12, 24, 36, 48, 60, 72, 6, 18, 30, 42],
                "Monthly Charges ($)": [65, 80, 45, 95, 110, 115, 55, 70, 85, 100],
                "Churn Risk": ["High", "Low", "Low", "High", "Low", "Low", "High", "High", "Low", "Low"],
            }
        )
        fig1 = px.scatter(
            mock_data,
            x="Tenure (Months)",
            y="Monthly Charges ($)",
            color="Churn Risk",
            color_discrete_map={"High": "#EF4444", "Low": "#10B981"},
            template="plotly_white",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Churn Risk Segmentation")
        risk_segments = pd.DataFrame(
            {"Segment": ["High Risk (>75%)", "Medium Risk (25-75%)", "Low Risk (<25%)"], "Count": [150, 400, 700]}
        )
        fig2 = px.pie(
            risk_segments,
            names="Segment",
            values="Count",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"],
            hole=0.4,
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict Customer Churn":
    st.header("Real-time Churn & LTV Scoring")

    # Input form
    with st.form("customer_input_form"):
        st.subheader("Customer Characteristics")
        c1, c2 = st.columns(2)
        with c1:
            customer_id = st.text_input("Customer ID", value="CUST-1049")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=79.95)
            total = st.number_input("Total Charges ($)", min_value=0.0, value=959.40)
        with c2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        submit = st.form_submit_path = st.form_submit_button("Run Analytics Engine")

    if submit:
        # Mock prediction output mimicking FastAPI response
        st.markdown("### Scoring Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.success("Analysis Complete!")
            st.metric("Churn Probability", "72.4%", delta="Critical Risk", delta_color="inverse")
            st.metric("Predicted LTV", "$1,040.00")
        with res_col2:
            st.warning("Recommended Actions:")
            st.markdown(
                """
                - 🚨 **High Risk Customer**: Urgent outreach recommended.
                - **Tactic 1**: Offer 15% discount code on a 1-year contract migration.
                - **Tactic 2**: Upgrade to premium tech support package for free.
                """
            )

else:
    st.header("Model Performance & SHAP Explainability")
    st.write("SHAP (SHapley Additive exPlanations) values indicate the attribution weights of each feature.")

    # Mock SHAP waterfall plot
    features = [
        "Contract: Month-to-month",
        "Tenure: 12 Months",
        "Internet Service: Fiber optic",
        "Monthly Charges: $79.95",
        "Tech Support: No",
    ]
    shap_values = [0.24, -0.15, 0.18, 0.08, 0.11]

    fig = go.Figure(
        go.Bar(
            x=shap_values,
            y=features,
            orientation="h",
            marker=dict(color=["#EF4444" if val > 0 else "#10B981" for val in shap_values]),
        )
    )
    fig.update_layout(
        title="SHAP Feature Impact (Positive increases churn probability)",
        xaxis_title="SHAP Value (Impact)",
        yaxis_title="Features",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
