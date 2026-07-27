"""
Unit tests for configuration loaders.

Validates that environment variables are read, defaulted, and translated into settings.
"""

from backend.core.settings import settings
from backend.core import config


def test_settings_load_defaults() -> None:
    """
    Test that default settings parameters are populated correctly.
    """
    assert settings.APP_VERSION == "1.0.0"
    assert settings.ENV == "testing"
    assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_database_url_generation() -> None:
    """
    Test that test database connection strings evaluate to local SQLite.
    """
    db_url = settings.get_db_url()
    assert "sqlite" in db_url


def test_core_constants() -> None:
    """
    Test that core application configuration constants are set.
    """
    assert config.API_TITLE != ""
    assert len(config.CORS_ORIGINS) > 0
    assert "random_state" in config.MODEL_DEFAULTS
