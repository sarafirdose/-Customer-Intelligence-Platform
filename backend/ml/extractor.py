"""
ETL Ingestion - Extraction Stage (Data Extractor).

Handles automated dataset downloads from Kaggle (if credentials exist)
or falls back to manual CSV file detection under data/raw/.
"""

import os
import zipfile
from pathlib import Path
from backend.core.logger import logger


class DataExtractor:
    """
    Handles retrieval and verification of the raw customer dataset.
    """

    def __init__(self, data_dir: str = "data/raw") -> None:
        """
        Initialize the DataExtractor.

        Args:
            data_dir: Target directory to search/save raw files.
        """
        self.data_dir = Path(data_dir)
        self.target_csv = self.data_dir / "telco_customer_churn.csv"
        self.zip_path = self.data_dir / "telco-customer-churn.zip"

    def is_dataset_available(self) -> bool:
        """
        Check if the raw customer CSV file is already present.

        Returns:
            bool: True if dataset is available.
        """
        return self.target_csv.exists()

    def download_from_kaggle(self) -> bool:
        """
        Attempt to download dataset using Kaggle API Python library or CLI.

        Returns:
            bool: True if download was successful, False otherwise.
        """
        # Check Kaggle credentials in env
        has_env_creds = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
        has_file_creds = Path("~/.kaggle/kaggle.json").expanduser().exists()

        if not (has_env_creds or has_file_creds):
            logger.info("Kaggle credentials not detected in environment or ~/.kaggle/. Skipping Kaggle API download.")
            return False

        logger.info("Kaggle credentials detected. Attempting automatic download via Kaggle API...")
        os.makedirs(self.data_dir, exist_ok=True)

        try:
            # Try running command or importing kaggle API
            import kaggle
            kaggle.api.authenticate()
            logger.info("Kaggle authentication successful. Downloading blastchar/telco-customer-churn...")
            kaggle.api.dataset_download_files("blastchar/telco-customer-churn", path=str(self.data_dir), unzip=False)
            logger.info(f"Dataset downloaded to {self.zip_path}")
            return True
        except Exception as e:
            logger.error(f"Failed download via Kaggle API library: {e}. Trying CLI fallback...")
            try:
                import subprocess
                subprocess.run(
                    ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn", "-p", str(self.data_dir)],
                    check=True,
                )
                logger.info("Kaggle CLI download completed successfully.")
                return True
            except Exception as cli_err:
                logger.error(f"Kaggle CLI download failed: {cli_err}")
                return False

    def unzip_dataset(self) -> bool:
        """
        Extract downloaded zip file and rename the raw CSV.

        Returns:
            bool: True if unzipped successfully.
        """
        if not self.zip_path.exists():
            return False

        logger.info(f"Extracting zip archive: {self.zip_path}...")
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Find extracted file (standard is WA_Fn-UseC_-Telco-Customer-Churn.csv)
            for f in self.data_dir.iterdir():
                if f.name.endswith(".csv") and f.name != "telco_customer_churn.csv":
                    # Rename to standard file name
                    if self.target_csv.exists():
                        os.remove(self.target_csv)
                    f.rename(self.target_csv)
                    logger.info(f"Extracted and renamed dataset file to: {self.target_csv}")
                    break

            # Cleanup zip file
            os.remove(self.zip_path)
            return True
        except Exception as e:
            logger.error(f"Error unzipping dataset archive: {e}")
            return False

    def extract(self) -> Path:
        """
        Execute the full extraction stage.

        Skips download if CSV is already present. Attempts Kaggle API if credentials
        are available, otherwise verifies that manual CSV exists.

        Returns:
            Path: Path to the target CSV file.

        Raises:
            FileNotFoundError: If the dataset is unavailable and cannot be downloaded.
        """
        logger.info("Extract stage: Starting raw dataset lookup.")

        if self.is_dataset_available():
            logger.info(f"Extract stage: Raw dataset already exists at {self.target_csv}. Skipping download.")
            return self.target_csv

        # Try automatic download
        download_success = self.download_from_kaggle()
        if download_success:
            self.unzip_dataset()

        if self.is_dataset_available():
            logger.info("Extract stage: Dataset successfully prepared.")
            return self.target_csv

        # If not downloaded automatically, search for manual files in the directory
        logger.info("Searching for manually placed CSV files in raw folder...")
        for f in self.data_dir.glob("*.csv"):
            if f.name != "telco_customer_churn.csv":
                # Rename it to our standard name
                f.rename(self.target_csv)
                logger.info(f"Detected manually placed file. Renamed {f.name} to {self.target_csv.name}")
                return self.target_csv

        # Final check
        if not self.is_dataset_available():
            error_msg = (
                "Dataset extraction failed: No dataset found at data/raw/telco_customer_churn.csv "
                "and Kaggle credentials were not available for automated download."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return self.target_csv
