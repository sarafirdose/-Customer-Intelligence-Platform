"""
SQLAlchemy database models for caching machine learning predictions.

Defines tables for Churn probabilities, LTV valuations, and retention suggestions.
"""

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from backend.database.models import Base


class Prediction(Base):
    """
    Model caching customer churn probability outcomes.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), index=True, nullable=False)
    churn_probability = Column(Float, nullable=False)
    is_churn = Column(Boolean, nullable=False)
    model_version = Column(String(50), nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Prediction customer='{self.customer_id}' probability={self.churn_probability} version='{self.model_version}'>"


class LtvPrediction(Base):
    """
    Model caching customer lifetime value (LTV) estimations.
    """

    __tablename__ = "ltv_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), index=True, nullable=False)
    predicted_ltv = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<LtvPrediction customer='{self.customer_id}' ltv={self.predicted_ltv} version='{self.model_version}'>"


class Recommendation(Base):
    """
    Model caching recommended retention marketing options.
    """

    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id", ondelete="CASCADE"), index=True, nullable=False)
    strategy = Column(String(255), nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Recommendation customer='{self.customer_id}' strategy='{self.strategy}'>"
