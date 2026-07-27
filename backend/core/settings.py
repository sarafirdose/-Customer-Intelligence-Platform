"""
Application settings configuration module.

This module defines the Settings class which validates and loads application
configuration from environment variables using Pydantic Settings.
"""

import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings class.

    Attributes:
        ENV: Application environment (development, testing, staging, production).
        DEBUG: Boolean indicating if debugging features should be enabled.
        APP_VERSION: Current API and project version.
        API_HOST: Network host to bind the API server to.
        API_PORT: Network port to bind the API server to.
        SECRET_KEY: Key used for secure operations like session signing.
        DB_HOST: Host name for PostgreSQL.
        DB_PORT: Port for PostgreSQL.
        DB_USER: Username for PostgreSQL.
        DB_PASSWORD: Password for PostgreSQL.
        DB_NAME: Database name.
        DB_SSL_MODE: SSL connectivity parameter for PostgreSQL.
        LOG_LEVEL: Logging level threshold.
        LOG_DIR: Directory where log files are stored.
        LOG_FILE_NAME: Base log file name.
    """

    ENV: Literal["development", "testing", "staging", "production"] = "development"
    DEBUG: bool = True
    APP_VERSION: str = "1.0.0"

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SECRET_KEY: str = "dev_secret_key_change_in_production_env"

    # Database Settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "cip_user"
    DB_PASSWORD: str = "cip_secure_password"
    DB_NAME: str = "customer_intelligence"
    DB_SSL_MODE: str = "disable"

    # Logging Settings
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "app.log"

    # Allow loading from environment variables/files
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_db_url(self) -> str:
        """
        Generate the database connection string.

        Supports PostgreSQL as the main production engine, and provides a local SQLite
        fallback for unit testing or simple development if specified in environment.

        Returns:
            str: Connection URI string.
        """
        # If in testing mode or if SQLite fallback is explicitly requested, we can use SQLite
        if self.ENV == "testing" and self.DB_HOST == "localhost" and os.environ.get("USE_SQLITE_TEST", "false").lower() == "true":
            return "sqlite:///./test.db"

        # Standard PostgreSQL connection string
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


# Instantiate settings instance for application-wide imports
settings = Settings()
