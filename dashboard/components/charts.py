"""
Plotly Interactive Charts Component for Streamlit Dashboard.

Defines interactive data visualizations with clean layout margins and non-overlapping legends.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_ltv_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of predicted LTV values with summary statistics.
    """
    ltv_col = "predicted_ltv" if "predicted_ltv" in df.columns else df.columns[0]
    ltvs = df[ltv_col].dropna()
    mean_val = float(ltvs.mean()) if not ltvs.empty else 0.0
    median_val = float(ltvs.median()) if not ltvs.empty else 0.0

    fig = px.histogram(
        df,
        x=ltv_col,
        nbins=10,
        title="Subscriber Lifetime Value (LTV) Distribution",
        labels={ltv_col: "Projected LTV ($)", "count": "Number of Subscribers"},
        color_discrete_sequence=["#8B5CF6"],
        template="plotly_dark",
        opacity=0.85,
    )
    fig.update_traces(
        marker_line_color="#1E1B2E",
        marker_line_width=1.5,
        hovertemplate="<b>LTV Bin Range</b>: $%{x}<br><b>Subscriber Count</b>: %{y}<extra></extra>",
    )
    if not ltvs.empty:
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#FFD700",
            line_width=2,
            annotation_text=f"Mean: ${mean_val:,.0f}",
            annotation_position="top right",
            annotation_font=dict(color="#FFD700", size=12),
        )
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="#00F0FF",
            line_width=2,
            annotation_text=f"Median: ${median_val:,.0f}",
            annotation_position="top left",
            annotation_font=dict(color="#00F0FF", size=12),
        )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        bargap=0.08,
        xaxis=dict(title="Subscriber Lifetime Value ($)"),
        yaxis=dict(title="Number of Subscribers"),
    )
    return fig


def plot_segment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of customer segments.
    """
    counts = df["customer_segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.pie(
        counts,
        names="Segment",
        values="Count",
        hole=0.4,
        title="Customer Segments Share",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def plot_cluster_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Tenure vs Monthly Charges colored by segment.
    """
    x_col = "tenure_months" if "tenure_months" in df.columns else ("churn_probability" if "churn_probability" in df.columns else df.columns[0])
    y_col = "monthly_charges" if "monthly_charges" in df.columns else ("predicted_ltv" if "predicted_ltv" in df.columns else df.columns[1])
    color_col = "customer_segment" if "customer_segment" in df.columns else ("segment" if "segment" in df.columns else None)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Customer Clusters ({x_col} vs {y_col})",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_rfm_heatmap(df_rfm: pd.DataFrame) -> go.Figure:
    """
    Heatmap of mean monetary values across R and F scores.
    """
    if df_rfm.empty:
        df_rfm = pd.DataFrame(columns=["R_score", "F_score", "M_score"])
    
    pivot = df_rfm.groupby(["R_score", "F_score"])["M_score"].mean().unstack().fillna(0)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"F-{i}" for i in pivot.columns],
            y=[f"R-{i}" for i in pivot.index],
            colorscale="Purples",
            colorbar=dict(title="Monetary (M)"),
        ),
        layout=go.Layout(
            title="RFM Grid: Mean Monetary Score by Recency & Frequency",
            template="plotly_dark",
        )
    )
    fig.update_layout(margin=dict(l=50, r=40, t=60, b=50))
    return fig


def plot_recommendation_chart(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of triggered recommendation campaigns.
    """
    counts = df.groupby(["primary_recommendation", "recommendation_priority"])["customer_id"].count().reset_index()
    counts.columns = ["Campaign", "Priority", "Volume"]
    
    fig = px.bar(
        counts,
        y="Campaign",
        x="Volume",
        color="Priority",
        orientation="h",
        title="Proactive Retention Campaigns Volume",
        labels={"Volume": "Customer Count", "Campaign": "Retention Offer"},
        color_discrete_map={"Critical": "#EF553B", "High": "#FECB52", "Medium": "#636EFA", "Low": "#00CC96"},
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=80, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of Subscriber Intelligence Scores with summary statistics.
    """
    score_col = "intelligence_score" if "intelligence_score" in df.columns else df.columns[0]
    scores = df[score_col].dropna()
    mean_val = float(scores.mean()) if not scores.empty else 0.0
    median_val = float(scores.median()) if not scores.empty else 0.0

    fig = px.histogram(
        df,
        x=score_col,
        nbins=10,
        title="Subscriber Intelligence Score (0–100) Distribution",
        labels={score_col: "Subscriber Score (0–100)", "count": "Number of Subscribers"},
        color_discrete_sequence=["#00CC96"],
        template="plotly_dark",
        opacity=0.85,
    )
    fig.update_traces(
        marker_line_color="#1E1B2E",
        marker_line_width=1.5,
        hovertemplate="<b>Score Range</b>: %{x}<br><b>Subscriber Count</b>: %{y}<extra></extra>",
    )
    if not scores.empty:
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#FFD700",
            line_width=2,
            annotation_text=f"Mean: {mean_val:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#FFD700", size=12),
        )
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="#00F0FF",
            line_width=2,
            annotation_text=f"Median: {median_val:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#00F0FF", size=12),
        )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        bargap=0.08,
        xaxis=dict(title="Subscriber Intelligence Score (0–100)", range=[0, 100]),
        yaxis=dict(title="Number of Subscribers"),
    )
    return fig


def plot_top_revenue_bar(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing top 10 revenue-generating accounts.
    """
    top_df = df.copy()
    val_col = "predicted_ltv" if "predicted_ltv" in top_df.columns else "total_charges"
    top_df = top_df.sort_values(by=val_col, ascending=False).head(10)
    
    fig = px.bar(
        top_df,
        x=val_col,
        y="customer_id",
        orientation="h",
        title="Top 10 Accounts by Customer Lifetime Value",
        labels={val_col: "Historical Spend ($)", "customer_id": "Customer Account ID"},
        color_discrete_sequence=["#FECB52"],
        template="plotly_dark",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=80, r=40, t=60, b=50))
    return fig


def plot_business_impact_bar(df: pd.DataFrame) -> go.Figure:
    """
    Targeted potential financial savings per priority tier.
    """
    impact = df.groupby("recommendation_priority")["estimated_revenue_saved"].sum().reset_index()
    impact.columns = ["Priority", "Saved Revenue"]
    
    fig = px.bar(
        impact,
        x="Priority",
        y="Saved Revenue",
        title="Saved Revenue Potential by Campaign Priority",
        labels={"Saved Revenue": "Revenue Saved ($)", "Priority": "Tier"},
        color="Priority",
        color_discrete_map={"Critical": "#EF553B", "High": "#FECB52", "Medium": "#636EFA", "Low": "#00CC96"},
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_segment_comparison_box(df: pd.DataFrame) -> go.Figure:
    """
    Boxplot showing churn risk probabilities binned by segment.
    """
    fig = px.box(
        df,
        x="customer_segment",
        y="churn_probability",
        points="all",
        title="Churn Probability spread by Customer Segment",
        labels={"customer_segment": "Segment", "churn_probability": "Churn Probability"},
        color="customer_segment",
        color_discrete_sequence=px.colors.qualitative.Safe,
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_revenue_distribution_box(df: pd.DataFrame) -> go.Figure:
    """
    Boxplot showing total charges spread binned by segment.
    """
    fig = px.box(
        df,
        x="customer_segment",
        y="predicted_ltv",
        title="Customer Lifetime Value distribution by Segment",
        labels={"customer_segment": "Segment", "predicted_ltv": "LTV Proxy ($)"},
        color="customer_segment",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
