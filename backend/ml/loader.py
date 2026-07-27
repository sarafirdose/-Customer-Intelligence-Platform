"""
ETL Ingestion - Load Stage (Data Loader).

Performs batched bulk insertions of customer ORM graphs into the database.
Enforces idempotency, counts metrics, and records logs in ImportHistory.
"""

from datetime import datetime
from typing import Any, Dict, List, Set
from sqlalchemy.orm import Session
from backend.core.logger import logger
from backend.models.customer import Customer
from backend.models.import_history import ImportHistory


class DataLoader:
    """
    Loads SQLAlchemy customer ORM objects into the database in batched transactions.
    """

    def __init__(self, db: Session, batch_size: int = 500) -> None:
        """
        Initialize the DataLoader.

        Args:
            db: Scoped SQLAlchemy database session.
            batch_size: Number of records to insert per batch chunk.
        """
        self.db = db
        self.batch_size = batch_size

    def _get_existing_customer_ids(self, customer_ids: List[str]) -> Set[str]:
        """
        Fetch customer_ids that already exist in the database.

        Args:
            customer_ids: List of alphanumeric customer IDs to inspect.

        Returns:
            Set[str]: Set of customer IDs that already exist.
        """
        if not customer_ids:
            return set()

        # Batch query existing IDs to prevent multiple single queries
        existing = (
            self.db.query(Customer.customer_id)
            .filter(Customer.customer_id.in_(customer_ids))
            .all()
        )
        return {row.customer_id for row in existing}

    def load(self, customers: List[Customer], filename: str) -> Dict[str, Any]:
        """
        Load ORM customer graphs into the database in batched chunks.

        Enforces idempotency by checking and filtering existing records.

        Args:
            customers: List of SQLAlchemy Customer objects.
            filename: Name of the raw file being ingested.

        Returns:
            Dict[str, Any]: Load metrics.
        """
        logger.info(f"Load stage: Initiating database load for {len(customers)} records.")
        started_at = datetime.utcnow()

        rows_processed = len(customers)
        rows_inserted = 0
        rows_skipped = 0
        rows_failed = 0

        # Collect customer IDs to run batch deduplication check
        customer_ids = [c.customer_id for c in customers]
        existing_ids = self._get_existing_customer_ids(customer_ids)
        logger.info(f"Load stage: Detected {len(existing_ids)} pre-existing customer records in database.")

        # Filter out duplicates
        pending_customers = []
        for c in customers:
            if c.customer_id in existing_ids:
                rows_skipped += 1
            else:
                pending_customers.append(c)

        # Batch write pending records
        total_pending = len(pending_customers)
        logger.info(f"Load stage: Loading {total_pending} new customer records in batches of {self.batch_size}.")

        try:
            for i in range(0, total_pending, self.batch_size):
                batch = pending_customers[i : i + self.batch_size]
                try:
                    self.db.add_all(batch)
                    self.db.commit()
                    rows_inserted += len(batch)
                except Exception as batch_err:
                    self.db.rollback()
                    logger.error(f"Failed to load batch indices {i} to {i + len(batch)}: {batch_err}")
                    rows_failed += len(batch)

            status = "success" if rows_failed == 0 else "degraded"
        except Exception as e:
            self.db.rollback()
            logger.error(f"Load stage: Critical load error encountered: {e}")
            status = "failed"
            rows_failed = total_pending - rows_inserted

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        logger.info(f"Load stage: DB loading finished in {duration:.2f}s. status='{status}'")

        # Save history log entry to database
        history_entry = ImportHistory(
            filename=filename,
            rows_processed=rows_processed,
            rows_inserted=rows_inserted,
            rows_skipped=rows_skipped,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
        )

        try:
            self.db.add(history_entry)
            self.db.commit()
            logger.info("Load stage: Ingestion metrics recorded in import_history table.")
        except Exception as history_err:
            self.db.rollback()
            logger.error(f"Failed to record import history log: {history_err}")

        return {
            "status": status,
            "rows_processed": rows_processed,
            "rows_inserted": rows_inserted,
            "rows_skipped": rows_skipped,
            "rows_failed": rows_failed,
            "execution_time_seconds": duration,
        }
