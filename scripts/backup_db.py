"""
Database Backup Automation Script.

Executes pg_dump against the active database configuration to save SQL snapshots
to backup folders.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from backend.core.logger import logger
from backend.core.settings import settings

BACKUP_DIR = Path(__file__).resolve().parents[1] / "data" / "backups"


def backup_database() -> None:
    """
    Perform a database backup using pg_dump.
    """
    if settings.get_db_url().startswith("sqlite"):
        logger.info("Using SQLite database. Copying file database to backups...")
        os.makedirs(BACKUP_DIR, exist_ok=True)
        sqlite_file = Path("test.db")
        if sqlite_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = BACKUP_DIR / f"test_backup_{timestamp}.db"
            import shutil
            shutil.copy(sqlite_file, dest)
            logger.info(f"Database backup succeeded. Saved to: {dest}")
        else:
            logger.warning("No SQLite database file found to backup.")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"{settings.DB_NAME}_backup_{timestamp}.sql"

    logger.info(f"Starting database backup. Saving to: {backup_file}")

    # Set password environment variable for pg_dump to prevent interactive prompts
    env = os.environ.copy()
    env["PGPASSWORD"] = settings.DB_PASSWORD

    command = [
        "pg_dump",
        "-h",
        settings.DB_HOST,
        "-p",
        str(settings.DB_PORT),
        "-U",
        settings.DB_USER,
        "-F",
        "c",  # Custom format (compressed binary format)
        "-b",  # Include large objects
        "-v",  # Verbose
        "-f",
        str(backup_file),
        settings.DB_NAME,
    ]

    try:
        result = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.info("Database backup completed successfully.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Database backup failed with exit code {e.returncode}")
        logger.error(e.stderr)
    except FileNotFoundError:
        logger.error(
            "pg_dump executable not found in path. Please install PostgreSQL client tools locally."
        )


if __name__ == "__main__":
    backup_database()
