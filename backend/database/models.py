"""
SQLAlchemy database models base declaration.

Defines the declarative Base class and basic shared models or mixins for the
Customer Intelligence Platform database schema.
"""

from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import DeclarativeBase, declared_attr


class Base(DeclarativeBase):
    """
    SQLAlchemy Declarative Base class.

    Provides common functionality and metadata hooks for all application models,
    including table name generation and dynamic serialization.
    """

    # Automatically generate table name from class name
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Derive table name automatically from lowercase class name.

        Returns:
            str: Plural/standardized lowercase table name.
        """
        name = cls.__name__.lower()
        if name.endswith("y"):
            return f"{name[:-1]}ies"
        elif name.endswith("s"):
            return name
        return f"{name}s"

    # Common audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize model columns to a standard Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary representing class column values.
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

