"""
SQLAlchemy connection engine and session management.

Exposes database engines, session creators, connection context dependencies,
and database validation utility functions.
"""

from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from backend.core.logger import logger
from backend.core.settings import settings

# Determine pool configurations depending on SQL Dialect (SQLite doesn't use pool size or max overflow)
db_url = settings.get_db_url()
engine_args = {}

if db_url.startswith("postgresql"):
    engine_args = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_pre_ping": True,  # Verifies connection viability before checkout
        "pool_recycle": 1800,  # Recycles connections every 30 minutes
    }
elif db_url.startswith("sqlite"):
    engine_args = {
        "connect_args": {"check_same_thread": False},
    }

# Create SQLAlchemy engine
engine = create_engine(db_url, **engine_args)

# Create SessionLocal factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency generator for database sessions.

    Used by FastAPI routers to inject database connections and guarantee cleanup
    after requests complete.

    Yields:
        Generator[Session, None, None]: Active SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_db_connection() -> bool:
    """
    Execute a mock statement (SELECT 1) to test database connection health.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    logger.info(f"Testing database connection on: {settings.DB_HOST}:{settings.DB_PORT}")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test succeeded.")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
    # Standard SQLite connection testing logic
