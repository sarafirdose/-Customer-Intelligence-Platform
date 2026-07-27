"""
Low-level direct database driver connection setup.

Provides direct access to raw PostgreSQL DBAPI connections for diagnostics
and compatibility with streaming utilities (e.g., loading large querysets).
"""

import contextlib
from typing import Generator
import psycopg2
from psycopg2.extensions import connection as PgConnection
from backend.core.logger import logger
from backend.core.settings import settings


@contextlib.contextmanager
def get_raw_connection() -> Generator[PgConnection, None, None]:
    """
    Context manager yielding a raw psycopg2 database connection.

    Yields:
        Generator[PgConnection, None, None]: Active raw DBAPI connection.

    Raises:
        ConnectionError: If connection cannot be established.
    """
    if settings.get_db_url().startswith("sqlite"):
        raise NotImplementedError("Raw psycopg2 connection is only supported for PostgreSQL.")

    conn = None
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            dbname=settings.DB_NAME,
            sslmode=settings.DB_SSL_MODE,
        )
        yield conn
    except Exception as e:
        logger.error(f"Raw psycopg2 connection failed: {e}")
        raise ConnectionError(f"Database connection error: {e}") from e
    finally:
        if conn is not None:
            conn.close()


def test_raw_connection() -> bool:
    """
    Test direct PostgreSQL connection viability.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    if settings.get_db_url().startswith("sqlite"):
        logger.warning("Local engine is configured for SQLite; skipping psycopg2 raw test.")
        return True

    try:
        with get_raw_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
        logger.info("Raw database driver connectivity check passed.")
        return True
    except Exception as e:
        logger.error(f"Raw database driver connectivity check failed: {e}")
        return False
