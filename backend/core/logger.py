"""
Application production logging configuration.

Creates console handlers, rotating file handlers for all events, and isolated
rotating file handlers for error logging. Ensures directories are created.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from backend.core.settings import settings


def setup_logging() -> logging.Logger:
    """
    Configure system-wide logging based on settings.

    Creates console, general app log, and error log handlers.

    Returns:
        logging.Logger: Root application logger instance.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Resolve log path
    log_dir = settings.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Standard formatter for structured log outputs
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear pre-existing handlers to prevent double logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Application General Rotating File Handler
    app_log_path = os.path.join(log_dir, settings.LOG_FILE_NAME)
    file_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. Dedicated Error Rotating File Handler
    error_log_path = os.path.join(log_dir, "error.log")
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Set external libraries log levels to prevent spam
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    logger = logging.getLogger("cip")
    logger.info("Logging configured successfully.")

    return logger


# Instantiate logger for imports
logger = setup_logging()
