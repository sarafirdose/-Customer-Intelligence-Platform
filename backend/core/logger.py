"""
Application production logging configuration.

Creates console handlers, rotating file handlers for all events, and isolated
rotating file handlers for error logging. Ensures directories are created.

Phase 6A Enhancement:
- Added JSONFormatter for structured JSON log emission (enabled via LOG_JSON=true)
- Added RequestIDFilter to inject correlation IDs into every log record
- RequestIDFilter reads from a ContextVar set per-request by logging middleware
"""

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from backend.core.settings import settings

# ---------------------------------------------------------------------------
# Request correlation ID context variable
# ---------------------------------------------------------------------------
# Each request sets this to a UUID; all log records within that request
# automatically include the same request_id for easy tracing.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    """Injects the current request_id ContextVar into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Emitted fields:
        timestamp, level, logger, filename, lineno,
        request_id, message, exc_info (if present)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """
    Configure system-wide logging based on settings.

    In production (LOG_JSON=true): emits JSON lines to console + files.
    In development (LOG_JSON=false): emits human-readable lines.

    Returns:
        logging.Logger: Named application logger instance ("cip").
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Resolve log path
    log_dir = settings.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Choose formatters based on environment
    if settings.LOG_JSON:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d]"
            " [req=%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    request_filter = RequestIDFilter()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear pre-existing handlers to prevent double logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)
    root_logger.addHandler(console_handler)

    # 2. Application General Rotating File Handler
    app_log_path = os.path.join(log_dir, settings.LOG_FILE_NAME)
    file_handler = RotatingFileHandler(
        app_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_filter)
    root_logger.addHandler(file_handler)

    # 3. Dedicated Error Rotating File Handler
    error_log_path = os.path.join(log_dir, "error.log")
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(request_filter)
    root_logger.addHandler(error_handler)

    # Suppress external library noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

    logger = logging.getLogger("cip")
    logger.info("Logging configured successfully.")

    return logger


# Instantiate logger for imports
logger = setup_logging()
