"""
Seed the Model Registry from existing artifact metadata files.

Run once after training to populate the registry:
    python scripts/seed_registry.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ml.registry import register_model, promote, list_all_models
from backend.core.logger import logger

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "models"


def seed_churn_model() -> None:
    """Register churn model from metadata.json."""
    meta_path = ARTIFACTS_DIR / "metadata.json"
    if not meta_path.exists():
        logger.warning("Churn metadata.json not found — skipping.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    entry = register_model(
        model_name="churn",
        version="v1.0.0",
        model_type="classifier",
        artifact_path="artifacts/models/best_model.pkl",
        metrics={
            "accuracy": meta.get("accuracy"),
            "roc_auc": meta.get("roc_auc"),
            "precision": meta.get("precision"),
            "recall": meta.get("recall"),
            "f1": meta.get("f1"),
            "brier_score": meta.get("brier_score"),
            "mean_latency_ms": meta.get("mean_latency_ms"),
        },
        dataset_rows=meta.get("dataset_rows", 7043),
        dataset_version="v1.0",
        feature_version="v1.0",
        git_commit=meta.get("git_commit_hash", "unknown"),
        python_version=meta.get("python_version", "unknown"),
        optimal_threshold=meta.get("optimal_threshold", 0.61),
        status="development",
        tags=["initial", "lgbm", "phase3"],
        notes="Initial churn model trained in Phase 3 on Telco dataset.",
    )
    promote("churn", "v1.0.0", "production")
    print(f"[OK] Registered churn@v1.0.0 -> production")


def seed_ltv_model() -> None:
    """Register LTV model from ltv_metadata.json."""
    meta_path = ARTIFACTS_DIR / "ltv_metadata.json"
    if not meta_path.exists():
        logger.warning("LTV metadata.json not found — skipping.")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    entry = register_model(
        model_name="ltv",
        version="v1.0.0",
        model_type="regressor",
        artifact_path="artifacts/models/ltv_model.pkl",
        metrics={
            "rmse": meta.get("rmse"),
            "mae": meta.get("mae"),
            "mape": meta.get("mape"),
            "r2": meta.get("r2"),
        },
        dataset_rows=7043,
        dataset_version="v1.0",
        feature_version="v1.0",
        git_commit="unknown",
        python_version="3.12.0",
        status="development",
        tags=["initial", "lightgbm", "phase4"],
        notes="Initial LTV regression model trained in Phase 4.",
    )
    promote("ltv", "v1.0.0", "production")
    print(f"[OK] Registered ltv@v1.0.0 -> production")


def seed_segmentation_model() -> None:
    """Register segmentation model (K-Means, no standard metrics)."""
    entry = register_model(
        model_name="segmentation",
        version="v1.0.0",
        model_type="clustering",
        artifact_path="artifacts/models/segmentation_model.pkl",
        metrics={
            "n_clusters": 3,
            "optimal_k_silhouette": "selected",
        },
        dataset_rows=7043,
        dataset_version="v1.0",
        feature_version="v1.0",
        git_commit="unknown",
        python_version="3.12.0",
        status="development",
        tags=["initial", "kmeans", "phase4"],
        notes="K-Means segmentation (k=3) trained in Phase 4.",
    )
    promote("segmentation", "v1.0.0", "production")
    print(f"[OK] Registered segmentation@v1.0.0 -> production")


if __name__ == "__main__":
    print("Seeding model registry from existing artifacts...\n")
    seed_churn_model()
    seed_ltv_model()
    seed_segmentation_model()

    registry = list_all_models()
    print(f"\nRegistry now contains {sum(len(v) for v in registry.values())} entries across {len(registry)} models.")
    print("Done.")
