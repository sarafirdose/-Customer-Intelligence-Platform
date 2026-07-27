"""
SQLAlchemy database model for customer profiles.

Defines the normalized Customer model and associations.
"""

from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from backend.database.models import Base


class Customer(Base):
    """
    Customer model capturing demographics and linking to billing, service, and contract tables.
    """

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), unique=True, index=True, nullable=False)

    # Demographics
    gender = Column(String(10), nullable=False)
    senior_citizen = Column(Integer, nullable=False)
    partner = Column(String(10), nullable=False)
    dependents = Column(String(10), nullable=False)
    tenure_months = Column(Integer, index=True, nullable=False)

    # Target
    churn = Column(Integer, index=True, nullable=False, default=0)

    # Foreign keys
    contract_id = Column(Integer, ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id", ondelete="CASCADE"), nullable=False)
    billing_id = Column(Integer, ForeignKey("billings.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    contract = relationship("Contract")
    service = relationship("Service")
    billing = relationship("Billing")

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Customer id={self.id} customer_id='{self.customer_id}' churn={self.churn}>"
