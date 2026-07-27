"""
Customer Lifetime Value (LTV) engine.

Uses customer financials, billing history, and churn curves to predict/estimate
the financial lifetime value of a customer profile.
"""

from backend.core.logger import logger


class LtvEngine:
    """
    Computes and predicts estimated Customer Lifetime Value.
    """

    def __init__(self, gross_margin: float = 0.70) -> None:
        """
        Initialize the LTV calculator.

        Args:
            gross_margin: Expected profit margins (0.0 to 1.0).
        """
        self.gross_margin = gross_margin

    def estimate_ltv(
        self, monthly_charges: float, churn_probability: float, tenure_months: int
    ) -> float:
        """
        Estimate customer LTV using retention probability and financial metrics.

        Standard simple formula:
        LTV = (Monthly Charges * Gross Margin) / Max(Churn Probability, Min Churn Threshold)

        Args:
            monthly_charges: Monthly subscription revenue.
            churn_probability: Model-predicted churn likelihood.
            tenure_months: Customer tenure in months.

        Returns:
            float: Estimated lifetime value in dollars.
        """
        logger.info(
            f"Estimating LTV: charges={monthly_charges}, churn={churn_probability}, tenure={tenure_months}"
        )

        # Establish floor churn rate to avoid division by zero
        adjusted_churn = max(churn_probability, 0.01)

        # Simple actuarial LTV calculation
        estimated_ltv = (monthly_charges * self.gross_margin) / adjusted_churn

        # Cap LTV based on realistic horizons (e.g. max 5 years/60 months of value)
        max_horizon_months = 60
        max_value = monthly_charges * self.gross_margin * max_horizon_months

        calculated_ltv = min(estimated_ltv, max_value)
        logger.info(f"LTV Estimation outcome: {calculated_ltv}")

        return round(calculated_ltv, 2)
