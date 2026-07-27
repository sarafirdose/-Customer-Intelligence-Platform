"""
SQLAlchemy database model for customer billing information.

Defines the Billing model.
"""

from sqlalchemy import Column, Float, Integer
from backend.database.models import Base


class Billing(Base):
    """
    Billing metrics class mapping charging rates.
    """

    id = Column(Integer, primary_key=True, autoincrement=True)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Billing id={self.id} monthly={self.monthly_charges} total={self.total_charges} >"
