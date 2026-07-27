"""
SQLAlchemy database model for customer contracts.

Defines the Contract model and mappings.
"""

from sqlalchemy import Column, Integer, String
from backend.database.models import Base


class Contract(Base):
    """
    Contract configuration class mapping payment and billing characteristics.
    """

    id = Column(Integer, primary_key=True, autoincrement=True)
    contract_type = Column(String(50), index=True, nullable=False)
    paperless_billing = Column(String(10), nullable=False)
    payment_method = Column(String(50), index=True, nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Contract id={self.id} type='{self.contract_type}' payment='{self.payment_method}'>"
