"""
ETL Ingestion - Profiling Stage (Data Profiler).

Profiles raw datasets before data cleaning, mapping shape, missingness ratios,
numerical statistics, and duplicate values to JSON reports.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from backend.core.logger import logger


class DataProfiler:
    """
    Profiles pandas DataFrames to summarize raw content distributions.
    """

    def __init__(self, reports_dir: str = "reports") -> None:
        """
        Initialize the DataProfiler.

        Args:
            reports_dir: Target directory to save profiling reports.
        """
        self.reports_dir = Path(reports_dir)
        os.makedirs(self.reports_dir, exist_ok=True)

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run profiling checks on the provided DataFrame and save findings.

        Args:
            df: Raw DataFrame to analyze.

        Returns:
            Dict[str, Any]: In-depth statistics dictionary.
        """
        logger.info("Profiling stage: Starting raw dataset analysis.")

        # Resolve numerical summaries safely
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_summaries = {}
        for col in numeric_cols:
            numeric_summaries[col] = {
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0,
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0.0,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0.0,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else 0.0,
                "missing_count": int(df[col].isnull().sum()),
            }

        # Resolve categorical value frequencies
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        categorical_summaries = {}
        for col in categorical_cols:
            # Handle list conversion cleanly
            val_counts = df[col].value_counts(dropna=False).head(10).to_dict()
            categorical_summaries[col] = {
                str(k): int(v) for k, v in val_counts.items()
            }

        # Calculate shape, duplicates, and missingness ratio
        shape = df.shape
        missing_count = int(df.isnull().sum().sum())
        total_cells = int(df.size)
        missing_percentage = float(missing_count / total_cells) if total_cells > 0 else 0.0

        # Handle customerID column duplicates check
        id_col = "customerID" if "customerID" in df.columns else (
            "customer_id" if "customer_id" in df.columns else None
        )
        duplicate_ids_count = 0
        if id_col is not None:
            duplicate_ids_count = int(df.duplicated(subset=[id_col]).sum())

        report = {
            "dataset_shape": {"rows": int(shape[0]), "columns": int(shape[1])},
            "cells_count": total_cells,
            "missing_values": {
                "total_missing": missing_count,
                "missing_percentage": missing_percentage,
                "missing_per_column": df.isnull().sum().to_dict(),
            },
            "duplicate_records": {
                "total_duplicates": int(df.duplicated().sum()),
                "duplicate_ids": duplicate_ids_count,
            },
            "numeric_summaries": numeric_summaries,
            "categorical_summaries": categorical_summaries,
        }

        # Save to reports/data_profile.json
        report_path = self.reports_dir / "data_profile.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Profiling report successfully saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write profiling report to disk: {e}")

        # Also write a simple Markdown overview for presentation
        md_path = self.reports_dir / "data_profile.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# Raw Dataset Profile Overview\n\n")
                f.write(f"- **Dimensions**: {shape[0]} rows, {shape[1]} columns\n")
                f.write(f"- **Total Missing Cells**: {missing_count} ({missing_percentage:.2%})\n")
                f.write(f"- **Duplicate Rows**: {report['duplicate_records']['total_duplicates']}\n")
                f.write(f"- **Duplicate Customer IDs**: {duplicate_ids_count}\n\n")
                f.write("## Numeric Summary\n")
                for col, stats in numeric_summaries.items():
                    f.write(f"### {col}\n")
                    f.write(f"- Mean: {stats['mean']:.2f}\n")
                    f.write(f"- Min/Max: {stats['min']:.2f} / {stats['max']:.2f}\n")
            logger.info(f"Profiling markdown overview saved to: {md_path}")
        except Exception as e:
            logger.error(f"Failed to write profiling md: {e}")

        return report
