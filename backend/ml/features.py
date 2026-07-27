"""
Feature engineering and data validation module.

Defines schemas using Pandera to enforce input quality constraints and
implements feature transformation steps (scaling, encoding) for modeling.
"""

from typing import Tuple
import pandas as pd
import pandera as pa
from backend.core.logger import logger

# Define Pandera validation schema for incoming customer training data
customer_schema = pa.DataFrameSchema(
    columns={
        "customer_id": pa.Column(str, required=True, nullable=False),
        "tenure_months": pa.Column(int, pa.Check.ge(0), required=True),
        "monthly_charges": pa.Column(float, pa.Check.ge(0.0), required=True),
        "total_charges": pa.Column(float, pa.Check.ge(0.0), required=True),
        "contract_type": pa.Column(str, required=True),
        "paperless_billing": pa.Column(str, pa.Check.isin(["Yes", "No"]), required=True),
        "internet_service": pa.Column(str, required=True),
        "tech_support": pa.Column(str, required=True),
    },
    coerce=True,
    strict=False,
)


class FeatureEngineer:
    """
    Manages data validation and feature engineering pipelines.
    """

    def __init__(self) -> None:
        """
        Initialize the Feature Engineer.
        """
        logger.info("Initializing Feature Engineering module.")

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input dataset against Pandera schemas.

        Args:
            df: Raw input DataFrame.

        Returns:
            pd.DataFrame: Validated (and coerced) DataFrame.

        Raises:
            SchemaError: If validation constraints fail.
        """
        logger.info("Validating dataset against schema rules.")
        try:
            validated_df = customer_schema.validate(df)
            logger.info("Dataset validation passed successfully.")
            return validated_df
        except pa.errors.SchemaError as e:
            logger.error(f"Dataset validation failed: {e}")
            raise e

    def build_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform encoding, scaling, and engineering of model features.

        Args:
            df: Input customer DataFrame.
            is_training: If True, target labels are returned alongside features.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and Target label series.
        """
        logger.info("Executing feature engineering pipeline.")

        # Validate schema first
        df_valid = self.validate_data(df)

        # Feature transformations (One-Hot Encoding, scaling placeholders)
        features = df_valid[
            [
                "tenure_months",
                "monthly_charges",
                "total_charges",
            ]
        ].copy()

        # Dummy one-hot mappings for contract type
        features["is_month_to_month"] = (df_valid["contract_type"] == "Month-to-month").astype(int)
        features["is_two_year"] = (df_valid["contract_type"] == "Two year").astype(int)
        features["has_tech_support"] = (df_valid["tech_support"] == "Yes").astype(int)

        # Extract target label if in training mode
        target = pd.Series()
        if is_training and "churn" in df_valid.columns:
            target = df_valid["churn"]
            logger.info("Extracted target labels from dataset.")

        return features, target
