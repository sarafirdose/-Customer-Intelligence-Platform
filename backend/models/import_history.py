"""
SQLAlchemy database model for tracking data import runs.

Defines the ImportHistory model.
"""

from sqlalchemy import Column, DateTime, Integer, String
from backend.database.models import Base


class ImportHistory(Base):
    """
    ImportHistory tracks ingestion pipelines execution runs.
    """

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    rows_processed = Column(Integer, nullable=False, default=0)
    rows_inserted = Column(Integer, nullable=False, default=0)
    rows_skipped = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)  # success, failed

    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"<ImportHistory id={self.id} file='{self.filename}' status='{self.status}' count={self.rows_inserted}>"
