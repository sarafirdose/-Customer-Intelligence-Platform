"""
CLI Drift Check Runner.

Usage:
    python scripts/run_drift_check.py --input reports/customer_intelligence.csv
    python scripts/run_drift_check.py --input data/new_customers.csv

Outputs:
    reports/drift_summary.json
    reports/feature_drift_report.md
    reports/drift/YYYY-MM-DD.json  (daily history)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from backend.ml.drift import run_drift_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature drift detection against a production CSV dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="reports/customer_intelligence.csv",
        help="Path to production dataset CSV (default: reports/customer_intelligence.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows.")

    print("Running drift analysis...")
    report = run_drift_check(df)

    severity = report["overall_severity"]
    severity_icons = {"Normal": "✅", "Warning": "⚠️", "Critical": "🚨"}
    icon = severity_icons.get(severity, "")

    print(f"\nOverall Drift Severity: {icon} {severity}")
    print("\nNumerical Feature PSI Scores:")
    for feat in report.get("numerical_drift", []):
        print(f"  {feat['feature']:<30} PSI={feat['psi']:.4f}  [{feat['severity']}]")

    print("\nCategorical Feature Max Shifts:")
    for feat in report.get("categorical_drift", []):
        print(f"  {feat['feature']:<30} MaxShift={feat['max_shift']:.4f}  [{feat['severity']}]")

    print("\nReports saved to:")
    print("  reports/drift_summary.json")
    print("  reports/feature_drift_report.md")
    print(f"  reports/drift/<today>.json")

    if severity == "Critical":
        print("\n🚨 ALERT: Critical drift detected. Consider retraining.")
        sys.exit(2)
    elif severity == "Warning":
        print("\n⚠️  WARNING: Moderate drift detected. Monitor closely.")


if __name__ == "__main__":
    main()
