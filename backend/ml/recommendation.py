"""
Retention recommendation engine.

Suggests strategic next-best-actions (discounts, contract renewals, upgrades)
by evaluating customer churn risk levels and lifetime value profiles.
"""

from typing import List
from backend.core.logger import logger


class RecommendationEngine:
    """
    Computes business retention recommendations for scored customer records.
    """

    def __init__(self, high_risk_threshold: float = 0.5) -> None:
        """
        Initialize the recommendation engine.

        Args:
            high_risk_threshold: Boundary probability identifying high risk.
        """
        self.high_risk_threshold = high_risk_threshold

    def get_retention_strategy(
        self, churn_probability: float, predicted_ltv: float
    ) -> List[str]:
        """
        Derive retention tactics by segmenting customers based on risk and value.

        Args:
            churn_probability: Model churn probability (0.0 to 1.0).
            predicted_ltv: Predicted customer lifetime value.

        Returns:
            List[str]: Set of action recommendations.
        """
        logger.info(
            f"Calculating recommendations: risk={churn_probability}, ltv={predicted_ltv}"
        )
        recommendations = []

        is_high_risk = churn_probability >= self.high_risk_threshold
        is_high_value = predicted_ltv >= 1000.0

        if is_high_risk and is_high_value:
            # VIP Churn Risk
            recommendations.append("Assign to VIP Dedicated Account Manager for outreach.")
            recommendations.append("Offer 20% loyalty discount on contract renewal.")
            recommendations.append("Provide complimentary premium tech support upgrade.")
        elif is_high_risk and not is_high_value:
            # Low Value Churn Risk
            recommendations.append("Trigger automated digital survey touchpoint.")
            recommendations.append("Suggest downgrade to lower-cost tier or contract option.")
        elif not is_high_risk and is_high_value:
            # VIP Safe Customer
            recommendations.append("Evaluate eligibility for cross-selling premium products.")
            recommendations.append("Enroll in VIP loyalty reward circle program.")
        else:
            # Safe, Normal Customer
            recommendations.append("Maintain standard digital engagement flow.")

        logger.info(f"Generated {len(recommendations)} retention options.")
        return recommendations
