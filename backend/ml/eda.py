"""
Exploratory Data Analysis (EDA) module.

Generates analytical summaries, performs statistical tests, and calculates
missingness rates and feature distributions across raw customer datasets.
"""

from typing import Dict, Any
import pandas as pd
from backend.core.logger import logger


class ExploratoryAnalysis:
    """
    Class providing EDA utilities for structured DataFrames.
    """

    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute general shape and statistical properties of a DataFrame.

        Args:
            df: Customer DataFrame.

        Returns:
            Dict[str, Any]: Metadata containing columns, statistics, and shape.
        """
        logger.info("Computing summary statistics for dataset.")
        summary = {
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1]),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        return summary

    @staticmethod
    def analyze_target_distribution(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Compute percentage distribution of the prediction target variable.

        Args:
            df: Customer DataFrame.
            target_col: Target column name (e.g. 'Churn').

        Returns:
            Dict[str, Any]: Mapping of classes to counts and ratios.
        """
        logger.info(f"Analyzing distribution of target column: {target_col}")
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not present in DataFrame.")
            raise ValueError(f"Target column '{target_col}' not found.")

        value_counts = df[target_col].value_counts().to_dict()
        total = sum(value_counts.values())

        return {
            str(cls): {"count": int(count), "percentage": float(count / total)}
            for cls, count in value_counts.items()
        }
