"""
ETL Data Ingestion Service.

Orchestrates the entire customer dataset ETL flow: Extracts raw CSV,
profiles the data, validates the schema using Pandera, cleans anomalies,
transforms tabular records to ORM graphs, and loads batches into PostgreSQL.
"""

from datetime import datetime
from typing import Any, Dict
import pandas as pd
from sqlalchemy.orm import Session

from backend.core.logger import logger
from backend.ml.extractor import DataExtractor
from backend.ml.profiling import DataProfiler
from backend.ml.validation import DataValidator
from backend.ml.cleaning import DataCleaner
from backend.ml.transformer import DataTransformer
from backend.ml.loader import DataLoader


class IngestionService:
    """
    Service class orchestrating the customer intelligence dataset ingestion pipeline.
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize IngestionService with dependencies.

        Args:
            db: Scoped SQLAlchemy database session.
        """
        self.db = db
        self.extractor = DataExtractor()
        self.profiler = DataProfiler()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.loader = DataLoader(db)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ETL data engineering ingestion pipeline.

        Returns:
            Dict[str, Any]: Summary metrics detailing the execution outcome.
        """
        start_time = datetime.utcnow()
        logger.info("ETL Pipeline: Starting customer dataset ingestion.")

        try:
            # 1. EXTRACT
            csv_path = self.extractor.extract()
            filename = csv_path.name
            df_raw = pd.read_csv(csv_path)
            logger.info(f"ETL Pipeline: Extracted {df_raw.shape[0]} rows from raw CSV.")

            # 2. PROFILE (pre-cleaning profile report)
            logger.info("ETL Pipeline: Generating pre-cleaning data profile report...")
            self.profiler.profile(df_raw)

            # 3. VALIDATE RAW
            logger.info("ETL Pipeline: Validating raw data schema...")
            is_raw_valid, raw_val_report = self.validator.validate_raw(df_raw)
            if not is_raw_valid:
                logger.warning(
                    f"ETL Pipeline: Raw data schema validation warnings present. "
                    f"Found {raw_val_report.get('failures_count', 0)} failure cases."
                )

            # 4. CLEAN
            logger.info("ETL Pipeline: Running data cleaning pipeline...")
            df_clean, clean_report = self.cleaner.clean(df_raw)

            # 5. VALIDATE CLEANED
            logger.info("ETL Pipeline: Validating cleaned data schema...")
            is_clean_valid, clean_val_report = self.validator.validate_clean(df_clean)
            if not is_clean_valid:
                logger.error(
                    f"ETL Pipeline: Cleaned dataset failed critical schema validation. "
                    f"Aborting database load. Errors count: {clean_val_report.get('failures_count', 0)}"
                )
                return {
                    "status": "failed",
                    "error": "Cleaned dataset failed Pandera schema validation.",
                    "details": clean_val_report,
                }

            # 6. TRANSFORM
            logger.info("ETL Pipeline: Transforming tabular records to database ORM models...")
            orm_customers = self.transformer.transform(df_clean)

            # 7. LOAD
            logger.info("ETL Pipeline: Loading records into database...")
            load_metrics = self.loader.load(orm_customers, filename)

            # Compile pipeline summary
            end_time = datetime.utcnow()
            pipeline_duration = (end_time - start_time).total_seconds()
            logger.info(f"ETL Pipeline: Ingestion complete. Duration: {pipeline_duration:.2f}s.")

            return {
                "status": "success",
                "filename": filename,
                "rows_processed": load_metrics["rows_processed"],
                "rows_inserted": load_metrics["rows_inserted"],
                "rows_skipped": load_metrics["rows_skipped"],
                "rows_failed": load_metrics["rows_failed"],
                "cleaning_summary": {
                    "dropped_rows": clean_report["dropped_rows"],
                    "duplicates_removed": clean_report["duplicates_removed"],
                    "missing_imputed": clean_report["missing_total_charges_imputed"],
                },
                "execution_time": f"{pipeline_duration:.2f}s",
            }

        except Exception as pipeline_err:
            logger.error(f"ETL Pipeline: Critical pipeline failure: {pipeline_err}")
            end_time = datetime.utcnow()
            pipeline_duration = (end_time - start_time).total_seconds()
            return {
                "status": "failed",
                "error": str(pipeline_err),
                "execution_time": f"{pipeline_duration:.2f}s",
            }
