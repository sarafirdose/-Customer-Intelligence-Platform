"""
SQLAlchemy database model for customer service options.

Defines the Service model and columns.
"""

from sqlalchemy import Column, Integer, String
from backend.database.models import Base


class Service(Base):
    """
    Service subscription class mapping communications, web, and entertainment items.
    """

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_service = Column(String(10), nullable=False)
    multiple_lines = Column(String(50), nullable=False)
    internet_service = Column(String(50), nullable=False)
    online_security = Column(String(50), nullable=False)
    online_backup = Column(String(50), nullable=False)
    device_protection = Column(String(50), nullable=False)
    tech_support = Column(String(50), nullable=False)
    streaming_tv = Column(String(50), nullable=False)
    streaming_movies = Column(String(50), nullable=False)

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<Service id={self.id} phone='{self.phone_service}' internet='{self.internet_service}'>"
