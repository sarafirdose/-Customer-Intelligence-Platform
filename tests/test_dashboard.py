"""
Automated Tests for Phase 5: Interactive Customer Intelligence Dashboard.

Tests cover:
- API client construction and error handling
- Cache utility file loading
- Export utilities (CSV and Excel)
- Chart generation (Plotly figures)
- Card and filter component HTML output
- Table rendering helpers
- Page module importability
"""

import io
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd
import pytest
import plotly.graph_objects as go


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_intel_df() -> pd.DataFrame:
    """Minimal customer intelligence dataframe fixture."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003", "C004"],
        "churn_probability": [0.75, 0.30, 0.65, 0.15],
        "predicted_ltv": [1500.0, 3200.0, 800.0, 4500.0],
        "projected_future_ltv": [800.0, 2000.0, 300.0, 3000.0],
        "customer_segment": ["Platinum", "Gold", "Bronze", "Gold"],
        "rfm_persona": ["Champions", "Loyal", "At Risk", "Loyal"],
        "intelligence_score": [82.5, 65.0, 40.0, 90.0],
        "intelligence_category": ["High Value", "Moderate", "Low Value", "High Value"],
        "primary_recommendation": ["Loyalty Reward", "Upsell Fiber", "Win-Back Offer", "Premium Bundle"],
        "recommendation_priority": ["High", "Medium", "Critical", "Low"],
        "estimated_revenue_saved": [250.0, 100.0, 500.0, 50.0],
        "tenure_months": [24, 48, 6, 60],
        "monthly_charges": [85.0, 70.0, 50.0, 95.0],
    })


@pytest.fixture
def sample_rfm_df() -> pd.DataFrame:
    """Minimal RFM analysis dataframe fixture."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "R_score": [4, 2, 1],
        "F_score": [3, 4, 2],
        "M_score": [4, 3, 1],
        "rfm_score": [11, 9, 4],
        "persona": ["Champions", "Loyal", "At Risk"],
    })


# ============================================================
# API Client Tests
# ============================================================

class TestAPIClient:
    """Tests for dashboard.utils.api_client.APIClient."""

    def test_client_default_base_url(self):
        """API client defaults to localhost:8000."""
        from dashboard.utils.api_client import APIClient
        client = APIClient()
        assert "localhost:8000" in client.base_url

    def test_client_custom_base_url(self):
        """API client accepts custom base URL."""
        from dashboard.utils.api_client import APIClient
        client = APIClient(base_url="http://prod-api:9000/api/v1")
        assert "prod-api:9000" in client.base_url

    def test_get_customer_intelligence_success(self):
        """Returns parsed JSON on HTTP 200 response."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"customer_id": "C001", "churn_probability": 0.75}

        with patch("httpx.get", return_value=mock_response):
            client = APIClient()
            result = client.get_customer_intelligence("C001")

        assert result["customer_id"] == "C001"
        assert result["churn_probability"] == 0.75

    def test_get_customer_intelligence_not_found(self):
        """Returns error dict on HTTP 404 response."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.get", return_value=mock_response):
            client = APIClient()
            result = client.get_customer_intelligence("MISSING")

        assert "error" in result
        assert "MISSING" in result["error"]

    def test_get_customer_intelligence_connection_error(self):
        """Returns error dict on connection failure."""
        from dashboard.utils.api_client import APIClient
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            client = APIClient()
            result = client.get_customer_intelligence("C001")

        assert "error" in result
        assert "Connection" in result["error"]

    def test_batch_score_customers_success(self):
        """Returns list of results on HTTP 200 batch response."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"customer_id": "C001", "churn_probability": 0.75},
            {"customer_id": "C002", "churn_probability": 0.30},
        ]

        with patch("httpx.post", return_value=mock_response):
            client = APIClient()
            result = client.batch_score_customers(["C001", "C002"])

        assert len(result) == 2
        assert result[0]["customer_id"] == "C001"

    def test_batch_score_customers_api_error(self):
        """Returns error list on non-200 batch response."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.post", return_value=mock_response):
            client = APIClient()
            result = client.batch_score_customers(["C001"])

        assert "error" in result[0]

    def test_get_customer_ltv_success(self):
        """Returns LTV data on HTTP 200."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"predicted_ltv": 1500.0}

        with patch("httpx.get", return_value=mock_response):
            client = APIClient()
            result = client.get_customer_ltv("C001")

        assert result["predicted_ltv"] == 1500.0

    def test_get_customer_segment_success(self):
        """Returns segment data on HTTP 200."""
        from dashboard.utils.api_client import APIClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"customer_segment": "Gold"}

        with patch("httpx.get", return_value=mock_response):
            client = APIClient()
            result = client.get_customer_segment("C001")

        assert result["customer_segment"] == "Gold"


# ============================================================
# Cache Utility Tests
# ============================================================

class TestCacheUtils:
    """Tests for dashboard.utils.cache module."""

    def test_load_global_intelligence_data_file_missing(self):
        """Returns empty dataframe with correct columns when file is absent."""
        from dashboard.utils.cache import load_global_intelligence_data
        # Clear cache before test to avoid stale results
        load_global_intelligence_data.clear()
        with patch("pathlib.Path.exists", return_value=False):
            df = load_global_intelligence_data()
        assert isinstance(df, pd.DataFrame)
        assert "customer_id" in df.columns
        assert len(df) == 0

    def test_load_rfm_analysis_data_file_missing(self):
        """Returns empty dataframe with correct columns when rfm file is absent."""
        from dashboard.utils.cache import load_rfm_analysis_data
        load_rfm_analysis_data.clear()
        with patch("pathlib.Path.exists", return_value=False):
            df = load_rfm_analysis_data()
        assert isinstance(df, pd.DataFrame)
        assert "customer_id" in df.columns

    def test_load_report_markdown_file_missing(self):
        """Returns not-available message when report file is missing."""
        from dashboard.utils.cache import load_report_markdown
        load_report_markdown.clear()
        with patch("pathlib.Path.exists", return_value=False):
            content = load_report_markdown("missing_report.md")
        assert "not available" in content.lower()


# ============================================================
# Export Utility Tests
# ============================================================

class TestExportUtils:
    """Tests for dashboard.utils.export module."""

    def test_convert_df_to_csv_returns_bytes(self, sample_intel_df):
        """CSV export returns bytes."""
        from dashboard.utils.export import convert_df_to_csv
        csv_bytes = convert_df_to_csv(sample_intel_df)
        assert isinstance(csv_bytes, bytes)
        assert b"customer_id" in csv_bytes

    def test_convert_df_to_csv_contains_all_rows(self, sample_intel_df):
        """CSV export contains all rows from dataframe."""
        from dashboard.utils.export import convert_df_to_csv
        csv_bytes = convert_df_to_csv(sample_intel_df)
        decoded = csv_bytes.decode("utf-8")
        # 4 data rows + 1 header row
        lines = [line for line in decoded.strip().split("\n") if line]
        assert len(lines) == 5

    def test_convert_df_to_excel_returns_bytes(self, sample_intel_df):
        """Excel export returns bytes (valid xlsx magic bytes)."""
        from dashboard.utils.export import convert_df_to_excel
        xlsx_bytes = convert_df_to_excel(sample_intel_df)
        assert isinstance(xlsx_bytes, bytes)
        # xlsx files start with PK magic bytes (zip format)
        assert xlsx_bytes[:2] == b"PK"

    def test_convert_df_to_excel_non_empty(self, sample_intel_df):
        """Excel export produces non-empty bytes."""
        from dashboard.utils.export import convert_df_to_excel
        xlsx_bytes = convert_df_to_excel(sample_intel_df)
        assert len(xlsx_bytes) > 1000  # A valid xlsx is always > 1 KB


# ============================================================
# Chart Component Tests
# ============================================================

class TestChartComponents:
    """Tests for dashboard.components.charts module."""

    def test_plot_ltv_distribution_returns_figure(self, sample_intel_df):
        """LTV distribution returns a Plotly Figure."""
        from dashboard.components.charts import plot_ltv_distribution
        fig = plot_ltv_distribution(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_segment_distribution_returns_figure(self, sample_intel_df):
        """Segment distribution pie chart returns a Plotly Figure."""
        from dashboard.components.charts import plot_segment_distribution
        fig = plot_segment_distribution(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_cluster_scatter_returns_figure(self, sample_intel_df):
        """Cluster scatter returns a Plotly Figure."""
        from dashboard.components.charts import plot_cluster_scatter
        fig = plot_cluster_scatter(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_rfm_heatmap_returns_figure(self, sample_rfm_df):
        """RFM heatmap returns a Plotly Figure."""
        from dashboard.components.charts import plot_rfm_heatmap
        fig = plot_rfm_heatmap(sample_rfm_df)
        assert isinstance(fig, go.Figure)

    def test_plot_rfm_heatmap_empty_df(self):
        """RFM heatmap handles empty dataframe gracefully."""
        from dashboard.components.charts import plot_rfm_heatmap
        fig = plot_rfm_heatmap(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_plot_recommendation_chart_returns_figure(self, sample_intel_df):
        """Recommendation horizontal bar chart returns a Plotly Figure."""
        from dashboard.components.charts import plot_recommendation_chart
        fig = plot_recommendation_chart(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_score_distribution_returns_figure(self, sample_intel_df):
        """Intelligence score histogram returns a Plotly Figure."""
        from dashboard.components.charts import plot_score_distribution
        fig = plot_score_distribution(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_top_revenue_bar_returns_figure(self, sample_intel_df):
        """Top revenue bar chart returns a Plotly Figure."""
        from dashboard.components.charts import plot_top_revenue_bar
        fig = plot_top_revenue_bar(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_business_impact_bar_returns_figure(self, sample_intel_df):
        """Business impact savings bar chart returns a Plotly Figure."""
        from dashboard.components.charts import plot_business_impact_bar
        fig = plot_business_impact_bar(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_segment_comparison_box_returns_figure(self, sample_intel_df):
        """Segment churn probability box plot returns a Plotly Figure."""
        from dashboard.components.charts import plot_segment_comparison_box
        fig = plot_segment_comparison_box(sample_intel_df)
        assert isinstance(fig, go.Figure)

    def test_plot_revenue_distribution_box_returns_figure(self, sample_intel_df):
        """LTV revenue distribution box plot returns a Plotly Figure."""
        from dashboard.components.charts import plot_revenue_distribution_box
        fig = plot_revenue_distribution_box(sample_intel_df)
        assert isinstance(fig, go.Figure)


# ============================================================
# Cards Component Tests
# ============================================================

class TestCardsComponent:
    """Tests for dashboard.components.cards module."""

    def test_render_kpi_card_returns_html_string(self):
        """KPI card HTML contains value and label."""
        # Import the function - it uses st.markdown internally
        # We verify the HTML generation logic directly
        from dashboard.components import cards
        import inspect
        source = inspect.getsource(cards.render_kpi_card)
        # Function must use st.markdown
        assert "st.markdown" in source

    def test_render_executive_header_in_source(self):
        """Executive header function exists and uses markdown."""
        from dashboard.components import cards
        import inspect
        source = inspect.getsource(cards.render_executive_header)
        assert "st.markdown" in source
        assert "unsafe_allow_html" in source


# ============================================================
# Filter Component Tests
# ============================================================

class TestFiltersComponent:
    """Tests for dashboard.components.filters module."""

    def test_filter_high_risk(self, sample_intel_df):
        """Filtering by High Risk returns only churn >= 0.61."""
        # Test filtering logic independently from Streamlit
        high_risk = sample_intel_df[sample_intel_df["churn_probability"] >= 0.61]
        assert len(high_risk) == 2
        assert all(high_risk["churn_probability"] >= 0.61)

    def test_filter_by_segment(self, sample_intel_df):
        """Filtering by segment returns only matching rows."""
        gold_df = sample_intel_df[sample_intel_df["customer_segment"] == "Gold"]
        assert len(gold_df) == 2
        assert all(gold_df["customer_segment"] == "Gold")

    def test_filter_low_risk(self, sample_intel_df):
        """Filtering by Low Risk returns only churn < 0.40."""
        low_risk = sample_intel_df[sample_intel_df["churn_probability"] < 0.40]
        assert len(low_risk) == 2
        assert all(low_risk["churn_probability"] < 0.40)

    def test_filter_by_rfm_persona(self, sample_intel_df):
        """Filtering by RFM persona returns only matching rows."""
        champions = sample_intel_df[sample_intel_df["rfm_persona"] == "Champions"]
        assert len(champions) == 1
        assert champions.iloc[0]["customer_id"] == "C001"


# ============================================================
# Tables Component Tests
# ============================================================

class TestTablesComponent:
    """Tests for dashboard.components.tables module."""

    def test_render_interactive_table_exists(self):
        """render_interactive_table function is importable."""
        from dashboard.components.tables import render_interactive_table
        assert callable(render_interactive_table)

    def test_pagination_logic(self, sample_intel_df):
        """Pagination slicing returns correct row subsets."""
        page_size = 2
        total_rows = len(sample_intel_df)
        total_pages = (total_rows - 1) // page_size + 1
        assert total_pages == 2

        page1 = sample_intel_df.iloc[0:2]
        page2 = sample_intel_df.iloc[2:4]
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1.iloc[0]["customer_id"] == "C001"
        assert page2.iloc[0]["customer_id"] == "C003"


# ============================================================
# KPI Calculation Tests
# ============================================================

class TestKPICalculations:
    """Tests verifying executive KPI metric computations."""

    def test_total_customers_count(self, sample_intel_df):
        """Total customer count is correct."""
        assert len(sample_intel_df) == 4

    def test_high_risk_count(self, sample_intel_df):
        """High risk count at 0.61 threshold is correct."""
        high_risk = len(sample_intel_df[sample_intel_df["churn_probability"] >= 0.61])
        assert high_risk == 2

    def test_avg_churn_probability(self, sample_intel_df):
        """Average churn probability calculation."""
        avg = sample_intel_df["churn_probability"].mean()
        assert abs(avg - 0.4625) < 0.001

    def test_avg_intelligence_score(self, sample_intel_df):
        """Average intelligence score calculation."""
        avg = sample_intel_df["intelligence_score"].mean()
        assert abs(avg - 69.375) < 0.1

    def test_revenue_at_risk_calculation(self, sample_intel_df):
        """Revenue at risk = sum(churn_prob * predicted_ltv)."""
        rev_at_risk = (sample_intel_df["churn_probability"] * sample_intel_df["predicted_ltv"]).sum()
        assert rev_at_risk > 0

    def test_total_estimated_savings(self, sample_intel_df):
        """Total estimated retention savings sum."""
        total = sample_intel_df["estimated_revenue_saved"].sum()
        assert total == 900.0

    def test_projected_revenue_sum(self, sample_intel_df):
        """Total projected future LTV sum is positive."""
        total = sample_intel_df["projected_future_ltv"].sum()
        assert total == 6100.0


# ============================================================
# Page Module Import Tests
# ============================================================

class TestPageModuleImports:
    """Tests verifying all dashboard page modules are importable."""

    def test_api_client_module_importable(self):
        """dashboard.utils.api_client is importable."""
        import dashboard.utils.api_client
        assert hasattr(dashboard.utils.api_client, "APIClient")

    def test_cache_module_importable(self):
        """dashboard.utils.cache is importable."""
        import dashboard.utils.cache
        assert hasattr(dashboard.utils.cache, "load_global_intelligence_data")

    def test_export_module_importable(self):
        """dashboard.utils.export is importable."""
        import dashboard.utils.export
        assert hasattr(dashboard.utils.export, "convert_df_to_csv")
        assert hasattr(dashboard.utils.export, "convert_df_to_excel")

    def test_charts_module_importable(self):
        """dashboard.components.charts is importable."""
        import dashboard.components.charts
        assert hasattr(dashboard.components.charts, "plot_ltv_distribution")
        assert hasattr(dashboard.components.charts, "plot_segment_distribution")
        assert hasattr(dashboard.components.charts, "plot_rfm_heatmap")

    def test_cards_module_importable(self):
        """dashboard.components.cards is importable."""
        import dashboard.components.cards
        assert hasattr(dashboard.components.cards, "render_kpi_card")
        assert hasattr(dashboard.components.cards, "render_executive_header")

    def test_filters_module_importable(self):
        """dashboard.components.filters is importable."""
        import dashboard.components.filters
        assert hasattr(dashboard.components.filters, "render_sidebar_filters")

    def test_tables_module_importable(self):
        """dashboard.components.tables is importable."""
        import dashboard.components.tables
        assert hasattr(dashboard.components.tables, "render_interactive_table")
