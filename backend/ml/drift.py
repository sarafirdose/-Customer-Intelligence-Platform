"""
Model Drift Detection Engine.

Compares production input distributions against training-time baselines using:
  - PSI (Population Stability Index) for numerical features
  - Chi-squared proportional shift for categorical features

Severity classification:
  - PSI < WARNING_THRESHOLD  → Normal
  - PSI < CRITICAL_THRESHOLD → Warning
  - PSI >= CRITICAL_THRESHOLD → Critical

Results are written to:
  - reports/drift/YYYY-MM-DD.json  (daily history)
  - reports/feature_drift_report.md (human-readable latest)
  - reports/drift_summary.json (machine-readable latest)
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.core.logger import logger
from backend.core.settings import settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
BASELINE_DIR = BASE_DIR / "artifacts" / "baseline"
REPORT_DIR = BASE_DIR / "reports"
DRIFT_HISTORY_DIR = BASE_DIR / settings.DRIFT_HISTORY_DIR

MEAN_PATH = BASELINE_DIR / "mean.json"
STD_PATH = BASELINE_DIR / "std.json"
CAT_DIST_PATH = BASELINE_DIR / "category_distribution.json"

WARNING_THRESHOLD = settings.DRIFT_WARNING_THRESHOLD
CRITICAL_THRESHOLD = settings.DRIFT_CRITICAL_THRESHOLD

# Numerical features tracked for drift
NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "charges_ratio",
    "total_services",
    "total_charges_log",
]

# Categorical features tracked for drift
CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "tenure_group",
]


# ---------------------------------------------------------------------------
# PSI Computation
# ---------------------------------------------------------------------------

def _compute_psi(
    baseline_values: np.ndarray,
    production_values: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between baseline and production arrays.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    Args:
        baseline_values: 1-D array of baseline (training) distribution.
        production_values: 1-D array of production (incoming) distribution.
        n_bins: Number of histogram bins.

    Returns:
        PSI score (float, >= 0).
    """
    # Compute bin edges from baseline
    breakpoints = np.percentile(baseline_values, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # de-duplicate identical percentiles

    def _bucket(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts = np.histogram(values, bins=edges)[0].astype(float)
        counts = np.where(counts == 0, 1e-6, counts)  # avoid log(0)
        return counts / counts.sum()

    pct_baseline = _bucket(baseline_values, breakpoints)
    pct_production = _bucket(production_values, breakpoints)

    psi = float(np.sum((pct_production - pct_baseline) * np.log(pct_production / pct_baseline)))
    return round(abs(psi), 6)


# ---------------------------------------------------------------------------
# Chi-squared proportional shift
# ---------------------------------------------------------------------------

def _compute_categorical_drift(
    baseline_dist: Dict[str, float],
    production_series: pd.Series,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Compare category proportions between baseline and production data.

    Returns:
        max_shift: Maximum absolute proportional shift across all categories.
        shift_table: Per-category {baseline, production, shift} dict.
    """
    prod_counts = production_series.value_counts(normalize=True)
    all_cats = set(list(baseline_dist.keys()) + list(prod_counts.index))

    shift_table: Dict[str, Dict[str, float]] = {}
    max_shift = 0.0

    for cat in all_cats:
        base_pct = baseline_dist.get(cat, 0.0)
        prod_pct = float(prod_counts.get(cat, 0.0))
        shift = abs(prod_pct - base_pct)
        shift_table[cat] = {
            "baseline_pct": round(base_pct, 4),
            "production_pct": round(prod_pct, 4),
            "shift": round(shift, 4),
        }
        max_shift = max(max_shift, shift)

    return round(max_shift, 6), shift_table


# ---------------------------------------------------------------------------
# Severity Classification
# ---------------------------------------------------------------------------

def _classify_psi(psi: float) -> str:
    """Map a PSI value to a severity label."""
    if psi < WARNING_THRESHOLD:
        return "Normal"
    elif psi < CRITICAL_THRESHOLD:
        return "Warning"
    return "Critical"


def _classify_shift(max_shift: float) -> str:
    """Map a max proportional shift to a severity label."""
    if max_shift < WARNING_THRESHOLD:
        return "Normal"
    elif max_shift < CRITICAL_THRESHOLD:
        return "Warning"
    return "Critical"


# ---------------------------------------------------------------------------
# Main drift check runner
# ---------------------------------------------------------------------------

def run_drift_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run full drift analysis on a production DataFrame.

    Args:
        df: Production data with the same feature columns as training.

    Returns:
        Full drift report dict.
    """
    if not MEAN_PATH.exists() or not CAT_DIST_PATH.exists():
        raise FileNotFoundError(
            "Baseline files not found. Run scripts/train_all.py to generate baselines."
        )

    with open(MEAN_PATH, "r") as f:
        baseline_means: Dict[str, float] = json.load(f)
    with open(STD_PATH, "r") as f:
        baseline_stds: Dict[str, float] = json.load(f)
    with open(CAT_DIST_PATH, "r") as f:
        baseline_cat: Dict[str, Dict[str, float]] = json.load(f)

    numerical_results: List[Dict[str, Any]] = []
    any_critical = False
    any_warning = False

    # --- Numerical drift (PSI) ---
    for feature in NUMERICAL_FEATURES:
        if feature not in df.columns:
            logger.warning(f"Drift: feature '{feature}' missing from production data — skipping.")
            continue

        prod_values = df[feature].dropna().values
        if len(prod_values) < 30:
            logger.warning(f"Drift: too few samples for '{feature}' ({len(prod_values)}) — skipping.")
            continue

        # Reconstruct baseline distribution using mean ± 3σ sampling
        mean = baseline_means.get(feature, 0.0)
        std = baseline_stds.get(feature, 1.0)
        rng = np.random.default_rng(seed=42)
        baseline_sample = rng.normal(loc=mean, scale=std, size=5000)

        psi = _compute_psi(baseline_sample, prod_values)
        severity = _classify_psi(psi)

        if severity == "Critical":
            any_critical = True
        elif severity == "Warning":
            any_warning = True

        numerical_results.append({
            "feature": feature,
            "psi": psi,
            "severity": severity,
            "baseline_mean": round(mean, 4),
            "production_mean": round(float(np.mean(prod_values)), 4),
            "baseline_std": round(std, 4),
            "production_std": round(float(np.std(prod_values)), 4),
        })

    # --- Categorical drift ---
    categorical_results: List[Dict[str, Any]] = []
    for feature in CATEGORICAL_FEATURES:
        if feature not in df.columns:
            continue
        if feature not in baseline_cat:
            continue

        max_shift, shift_table = _compute_categorical_drift(
            baseline_cat[feature], df[feature]
        )
        severity = _classify_shift(max_shift)

        if severity == "Critical":
            any_critical = True
        elif severity == "Warning":
            any_warning = True

        categorical_results.append({
            "feature": feature,
            "max_shift": max_shift,
            "severity": severity,
            "categories": shift_table,
        })

    # --- Overall severity ---
    if any_critical:
        overall_severity = "Critical"
    elif any_warning:
        overall_severity = "Warning"
    else:
        overall_severity = "Normal"

    report: Dict[str, Any] = {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_rows": len(df),
        "overall_severity": overall_severity,
        "psi_warning_threshold": WARNING_THRESHOLD,
        "psi_critical_threshold": CRITICAL_THRESHOLD,
        "numerical_drift": numerical_results,
        "categorical_drift": categorical_results,
    }

    _persist_drift_report(report)
    return report


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_drift_report(report: Dict[str, Any]) -> None:
    """Save drift report to history dir and update latest summary files."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # 1. Daily history entry
    history_path = DRIFT_HISTORY_DIR / f"{today}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 2. Latest summary JSON
    summary_path = REPORT_DIR / "drift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 3. Markdown report
    md_path = REPORT_DIR / "feature_drift_report.md"
    _write_markdown_report(report, md_path)

    logger.info(
        f"Drift report saved: overall={report['overall_severity']}, "
        f"history={history_path.name}"
    )


def _write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    """Render drift report as human-readable Markdown."""
    severity_emoji = {"Normal": "✅", "Warning": "⚠️", "Critical": "🚨"}

    lines = [
        "# Feature Drift Report",
        "",
        f"**Run Date**: {report['run_at']}",
        f"**Dataset Rows**: {report['dataset_rows']:,}",
        f"**Overall Severity**: {severity_emoji.get(report['overall_severity'], '')} {report['overall_severity']}",
        f"**PSI Warning Threshold**: {report['psi_warning_threshold']}",
        f"**PSI Critical Threshold**: {report['psi_critical_threshold']}",
        "",
        "---",
        "",
        "## Numerical Feature Drift (PSI)",
        "",
        "| Feature | PSI | Severity | Baseline μ | Prod μ | Baseline σ | Prod σ |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in report.get("numerical_drift", []):
        emoji = severity_emoji.get(r["severity"], "")
        lines.append(
            f"| {r['feature']} | {r['psi']:.4f} | {emoji} {r['severity']} "
            f"| {r['baseline_mean']} | {r['production_mean']} "
            f"| {r['baseline_std']} | {r['production_std']} |"
        )

    lines += ["", "---", "", "## Categorical Feature Drift (Max Proportional Shift)", ""]
    for r in report.get("categorical_drift", []):
        emoji = severity_emoji.get(r["severity"], "")
        lines.append(f"### {r['feature']} — {emoji} {r['severity']} (max shift: {r['max_shift']:.4f})")
        lines.append("")
        lines.append("| Category | Baseline % | Production % | Shift |")
        lines.append("|---|---|---|---|")
        for cat, vals in r.get("categories", {}).items():
            lines.append(
                f"| {cat} | {vals['baseline_pct']:.2%} "
                f"| {vals['production_pct']:.2%} | {vals['shift']:.4f} |"
            )
        lines.append("")

    lines += ["---", "", "*Generated automatically by drift.py*"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_drift_history() -> List[Dict[str, Any]]:
    """Return all saved drift reports sorted by date ascending."""
    if not DRIFT_HISTORY_DIR.exists():
        return []
    history = []
    for json_file in sorted(DRIFT_HISTORY_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_date"] = json_file.stem
                history.append(data)
        except Exception:
            pass
    return history
