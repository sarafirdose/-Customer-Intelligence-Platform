"""
ETL Ingestion - Cleaning Stage (Data Cleaner).

Cleans raw customer dataframes by trimming strings, parsing columns to snake_case,
casting data types, resolving null total charges, mapping churn flags, and
removing duplicate customer records.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from backend.core.logger import logger


class DataCleaner:
    """
    Cleans, casts, and normalizes raw Pandas DataFrames.
    """

    def __init__(self, reports_dir: str = "reports/validation") -> None:
        """
        Initialize the DataCleaner.

        Args:
            reports_dir: Folder to save cleaning summaries.
        """
        self.reports_dir = Path(reports_dir)
        os.makedirs(self.reports_dir, exist_ok=True)

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute cleaning operations on the raw DataFrame.

        Args:
            df: Input raw DataFrame.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and cleaning summary.
        """
        logger.info("Cleaning stage: Starting raw dataset cleansing.")

        # Create a deep copy to prevent modifications to original reference
        clean_df = df.copy()

        initial_rows = len(clean_df)

        # 1. Rename columns to standard database snake_case
        column_mapping = {
            "customerID": "customer_id",
            "gender": "gender",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "tenure": "tenure_months",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract_type",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn",
        }
        clean_df = clean_df.rename(columns=column_mapping)

        # 2. Trim whitespace from all string columns
        string_cols = clean_df.select_dtypes(include=["object"]).columns
        for col in string_cols:
            clean_df[col] = clean_df[col].astype(str).str.strip()

        # 3. Handle duplicate records
        duplicates_removed = int(clean_df.duplicated(subset=["customer_id"]).sum())
        if duplicates_removed > 0:
            clean_df = clean_df.drop_duplicates(subset=["customer_id"], keep="first")
            logger.info(f"Cleaning stage: Removed {duplicates_removed} duplicate customer records.")

        # 4. Data Type conversions and missing values resolution
        # Convert total_charges to float, coercing empty spaces to NaN
        clean_df["total_charges"] = pd.to_numeric(clean_df["total_charges"], errors="coerce")

        # In Telco Churn, missing total charges correspond exactly to tenure = 0.
        # Impute missing total charges to 0.0 where tenure is 0.
        missing_total_charges = int(clean_df["total_charges"].isnull().sum())
        imputed_count = 0
        if missing_total_charges > 0:
            zero_tenure_mask = (clean_df["tenure_months"] == 0) & (clean_df["total_charges"].isnull())
            imputed_count = int(zero_tenure_mask.sum())
            clean_df.loc[zero_tenure_mask, "total_charges"] = 0.0

            # If any remaining NaNs exist, impute with monthly charges (first month charges)
            remaining_nans_mask = clean_df["total_charges"].isnull()
            remaining_nans_count = int(remaining_nans_mask.sum())
            if remaining_nans_count > 0:
                clean_df.loc[remaining_nans_mask, "total_charges"] = clean_df.loc[remaining_nans_mask, "monthly_charges"]
                imputed_count += remaining_nans_count
            logger.info(f"Cleaning stage: Imputed {imputed_count} missing values in 'total_charges'.")

        # Convert Churn from "Yes"/"No" to 1/0
        clean_df["churn"] = clean_df["churn"].map({"Yes": 1, "No": 0})
        # If any rows had missing Churn, drop them
        invalid_churn_count = int(clean_df["churn"].isnull().sum())
        if invalid_churn_count > 0:
            clean_df = clean_df.dropna(subset=["churn"])
            clean_df["churn"] = clean_df["churn"].astype(int)
            logger.warning(f"Cleaning stage: Dropped {invalid_churn_count} records due to invalid/missing churn values.")

        # Force correct types
        clean_df["senior_citizen"] = clean_df["senior_citizen"].astype(int)
        clean_df["tenure_months"] = clean_df["tenure_months"].astype(int)
        clean_df["monthly_charges"] = clean_df["monthly_charges"].astype(float)
        clean_df["total_charges"] = clean_df["total_charges"].astype(float)

        final_rows = len(clean_df)
        dropped_rows = initial_rows - final_rows

        summary = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "dropped_rows": dropped_rows,
            "duplicates_removed": duplicates_removed,
            "missing_total_charges_imputed": imputed_count,
            "invalid_churn_dropped": invalid_churn_count,
            "columns_cleaned": list(clean_df.columns),
        }

        # Save to reports/validation/cleaning_report.json
        report_path = self.reports_dir / "cleaning_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Cleaning summary saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write cleaning report: {e}")

        return clean_df, summary
