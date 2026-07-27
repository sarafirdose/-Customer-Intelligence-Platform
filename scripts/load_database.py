"""
Database seeding and loader utility script.

Triggers the production ETL ingestion pipeline to clean and load raw customer data
into the database.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import logger
from backend.database.database import SessionLocal, engine
from backend.database.models import Base
from backend.services.ingestion_service import IngestionService


def load_database() -> None:
    """
    Execute the ingestion pipeline to load and validate customer data.
    """
    logger.info("Initializing database schema if not present...")
    Base.metadata.create_all(bind=engine)

    logger.info("Instantiating ETL pipeline for database seeding...")
    db = SessionLocal()
    try:
        service = IngestionService(db)
        metrics = service.run_pipeline()

        if metrics["status"] == "success":
            logger.info("Database seeding completed successfully.")
            logger.info(f"Summary: Processed={metrics['rows_processed']}, "
                        f"Inserted={metrics['rows_inserted']}, "
                        f"Skipped={metrics['rows_skipped']}, "
                        f"Duration={metrics['execution_time']}")
        else:
            logger.error(f"Database seeding failed: {metrics.get('error')}")
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    load_database()
