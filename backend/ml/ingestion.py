"""
Data ingestion module.

Provides classes and methods to import raw telemetry files, process streaming sources,
and prepare base structured DataFrames for EDA and ML training pipelines.
"""

from typing import Union
from pathlib import Path
import pandas as pd
from backend.core.logger import logger


class DataIngestion:
    """
    Handles ingestion of raw datasets into standard structured formats.
    """

    def __init__(self, raw_data_path: Union[str, Path]) -> None:
        """
        Initialize the ingestion engine.

        Args:
            raw_data_path: Path to raw dataset storage.
        """
        self.raw_data_path = Path(raw_data_path)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Read a CSV file from raw data storage and execute initial validation.

        Args:
            filename: Target file name to read.

        Returns:
            pd.DataFrame: Cleaned Pandas DataFrame.

        Raises:
            FileNotFoundError: If the target file is missing.
        """
        target_path = self.raw_data_path / filename
        logger.info(f"Loading raw dataset from: {target_path}")

        if not target_path.exists():
            error_msg = f"Raw CSV dataset not found at {target_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load dataset
            df = pd.read_csv(target_path)
            logger.info(f"Dataset ingested successfully. Dimensions: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to ingest raw dataset: {e}")
            raise
