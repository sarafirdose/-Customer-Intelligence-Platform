"""
ETL Ingestion - Validation Stage (Data Validator).

Applies Pandera schema rules to validate types, unique constraints, values,
and required columns. Compiles validation errors to JSON reports.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import pandera as pa
from backend.core.logger import logger


# 1. Raw Dataset Validation Schema
raw_schema = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(str, required=True),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"]), required=True),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), required=True),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "tenure": pa.Column(int, pa.Check.ge(0), required=True),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "MultipleLines": pa.Column(str, pa.Check.isin(["Yes", "No", "No phone service"]), required=True),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"]), required=True),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "OnlineBackup": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "DeviceProtection": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "TechSupport": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "StreamingTV": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "StreamingMovies": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"]), required=True),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "PaymentMethod": pa.Column(str, pa.Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]), required=True),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0.0), required=True),
        "TotalCharges": pa.Column(object, required=True),  # Can contain spaces, validated after typecasting in clean_schema
        "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
    },
    coerce=True,
    strict=False,
)

# 2. Cleaned Dataset Validation Schema (strictly typed, space-free)
clean_schema = pa.DataFrameSchema(
    columns={
        "customer_id": pa.Column(str, unique=True, required=True),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"]), required=True),
        "senior_citizen": pa.Column(int, pa.Check.isin([0, 1]), required=True),
        "partner": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "dependents": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "tenure_months": pa.Column(int, pa.Check.ge(0), required=True),
        "phone_service": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "multiple_lines": pa.Column(str, pa.Check.isin(["Yes", "No", "No phone service"]), required=True),
        "internet_service": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"]), required=True),
        "online_security": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "online_backup": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "device_protection": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "tech_support": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "streaming_tv": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "streaming_movies": pa.Column(str, pa.Check.isin(["Yes", "No", "No internet service"]), required=True),
        "contract_type": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"]), required=True),
        "paperless_billing": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "payment_method": pa.Column(str, pa.Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]), required=True),
        "monthly_charges": pa.Column(float, pa.Check.ge(0.0), required=True),
        "total_charges": pa.Column(float, pa.Check.ge(0.0), required=True),
        "churn": pa.Column(int, pa.Check.isin([0, 1]), required=True),
    },
    coerce=True,
    strict=True,
)


class DataValidator:
    """
    Applies Pandera schema rules to validate quality of customer records.
    """

    def __init__(self, reports_dir: str = "reports/validation") -> None:
        """
        Initialize the DataValidator.

        Args:
            reports_dir: Folder path where schema logs are written.
        """
        self.reports_dir = Path(reports_dir)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _compile_failure_report(self, err: pa.errors.SchemaErrors) -> Dict[str, Any]:
        """
        Extract validation failure logs from a Pandera exception.

        Args:
            err: SchemaErrors exception.

        Returns:
            Dict[str, Any]: Failure mapping.
        """
        failures = []
        # Check if failure cases exist
        if err.failure_cases is not None:
            for _, row in err.failure_cases.iterrows():
                failures.append({
                    "column": str(row.get("column", "N/A")),
                    "check": str(row.get("check", "N/A")),
                    "failure_value": str(row.get("failure_case", "N/A")),
                    "index": str(row.get("index", "N/A")),
                })

        return {
            "status": "failed",
            "error_message": str(err),
            "failures_count": len(failures),
            "failures": failures,
        }

    def validate_raw(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate raw pandas DataFrame structure.

        Args:
            df: Input raw customer DataFrame.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation_report)
        """
        logger.info("Validation stage: Starting raw dataset schema checks.")
        try:
            raw_schema.validate(df, lazy=True)
            report = {"status": "success", "failures_count": 0}
            self._save_report("raw_validation_report.json", report)
            logger.info("Validation stage: Raw schema check passed.")
            return True, report
        except pa.errors.SchemaErrors as e:
            logger.error("Validation stage: Raw schema validation failed.")
            report = self._compile_failure_report(e)
            self._save_report("raw_validation_report.json", report)
            return False, report

    def validate_clean(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate cleaned data DataFrame formats.

        Args:
            df: Cleaned and formatted DataFrame.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, validation_report)
        """
        logger.info("Validation stage: Starting cleaned dataset schema checks.")
        try:
            clean_schema.validate(df, lazy=True)
            report = {"status": "success", "failures_count": 0}
            self._save_report("validation_report.json", report)
            logger.info("Validation stage: Cleaned schema check passed.")
            return True, report
        except pa.errors.SchemaErrors as e:
            logger.error("Validation stage: Cleaned schema validation failed.")
            report = self._compile_failure_report(e)
            self._save_report("validation_report.json", report)
            return False, report

    def _save_report(self, filename: str, report: Dict[str, Any]) -> None:
        """
        Save validation output documents.

        Args:
            filename: Target file name to write.
            report: Validation dictionary details.
        """
        dest = self.reports_dir / filename
        try:
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to: {dest}")
        except Exception as e:
            logger.error(f"Failed to save validation report to {dest}: {e}")
