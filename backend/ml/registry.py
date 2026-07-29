"""
Local File-Based Model Registry.

Tracks all trained model versions with full metadata, supports promotion,
rollback, tagging, and side-by-side metric comparison.

FileLock ensures concurrent-safe writes; no database required.

Registry location: artifacts/registry/model_registry.json
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock, Timeout

from backend.core.logger import logger
from backend.core.settings import settings

# ---------------------------------------------------------------------------
# Registry paths
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(settings.REGISTRY_PATH)
REGISTRY_DIR = REGISTRY_PATH.parent
LOCK_PATH = REGISTRY_PATH.with_suffix(".lock")

# In-process lock guards the outer Python layer; FileLock guards the file
_in_process_lock = threading.Lock()

# Valid status transitions
VALID_STATUSES = {"development", "staging", "production", "archived"}


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def _ensure_registry_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _read_registry() -> Dict[str, List[Dict[str, Any]]]:
    """Load registry JSON; returns empty dict if file absent."""
    _ensure_registry_dir()
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Atomically write registry JSON using FileLock."""
    _ensure_registry_dir()
    try:
        with FileLock(str(LOCK_PATH), timeout=10):
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    except Timeout:
        logger.error("Registry FileLock timed out — concurrent write conflict.")
        raise RuntimeError("Model registry is locked by another process.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(
    model_name: str,
    version: str,
    model_type: str,
    artifact_path: str,
    metrics: Dict[str, Any],
    dataset_rows: int = 0,
    dataset_version: str = "v1.0",
    feature_version: str = "v1.0",
    git_commit: str = "unknown",
    python_version: str = "unknown",
    optimal_threshold: Optional[float] = None,
    status: str = "development",
    tags: Optional[List[str]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Register a new model version in the registry.

    Args:
        model_name: Logical name (e.g. 'churn', 'ltv', 'segmentation').
        version: Semantic version string (e.g. 'v1.0.0').
        model_type: Type descriptor ('classifier', 'regressor', 'clustering').
        artifact_path: Relative path to the saved .pkl file.
        metrics: Dict of evaluation metrics (roc_auc, rmse, etc.).
        dataset_rows: Number of training rows.
        dataset_version: Tag identifying the dataset snapshot.
        feature_version: Tag identifying the feature engineering version.
        git_commit: Git SHA at training time.
        python_version: Python version used.
        optimal_threshold: Decision threshold (classifiers only).
        status: Initial status ('development' | 'staging' | 'production').
        tags: Free-form string labels (e.g. ['stable', 'july-release']).
        notes: Free-text notes.

    Returns:
        The newly registered entry dict.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}.")

    entry: Dict[str, Any] = {
        "model_name": model_name,
        "version": version,
        "model_type": model_type,
        "artifact_path": artifact_path,
        "metrics": metrics,
        "dataset_rows": dataset_rows,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "git_commit": git_commit,
        "python_version": python_version,
        "optimal_threshold": optimal_threshold,
        "status": status,
        "tags": tags or [],
        "notes": notes,
        "registered_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    with _in_process_lock:
        registry = _read_registry()
        versions = registry.setdefault(model_name, [])

        # Prevent duplicate versions
        existing_versions = [e["version"] for e in versions]
        if version in existing_versions:
            logger.warning(f"Registry: version {version} of '{model_name}' already exists — skipping.")
            return next(e for e in versions if e["version"] == version)

        versions.append(entry)
        _write_registry(registry)

    logger.info(f"Registry: registered {model_name}@{version} [{status}]")
    return entry


def promote(model_name: str, version: str, new_status: str) -> Dict[str, Any]:
    """
    Change the status of a specific model version.

    When promoting to 'production', all other versions of the same model
    are demoted to 'staging'.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'.")

    with _in_process_lock:
        registry = _read_registry()
        versions = registry.get(model_name, [])

        target = next((e for e in versions if e["version"] == version), None)
        if target is None:
            raise KeyError(f"Model '{model_name}' version '{version}' not found in registry.")

        if new_status == "production":
            # Demote all existing production entries to staging
            for entry in versions:
                if entry["status"] == "production" and entry["version"] != version:
                    entry["status"] = "staging"
                    logger.info(f"Registry: demoted {model_name}@{entry['version']} → staging")

        target["status"] = new_status
        _write_registry(registry)

    logger.info(f"Registry: promoted {model_name}@{version} → {new_status}")
    return target


def rollback(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Roll back to the previous production model.

    Demotes current production version to 'staging' and promotes
    the most recent prior staging version to 'production'.

    Returns:
        The newly promoted entry, or None if rollback is not possible.
    """
    with _in_process_lock:
        registry = _read_registry()
        versions = registry.get(model_name, [])

        current_prod = next((e for e in versions if e["status"] == "production"), None)
        if current_prod is None:
            logger.warning(f"Registry: no production model found for '{model_name}' — nothing to roll back.")
            return None

        # Find most recent staging version (excluding current prod)
        staging_candidates = [
            e for e in versions
            if e["status"] == "staging" and e["version"] != current_prod["version"]
        ]
        staging_candidates.sort(key=lambda e: e["registered_at"], reverse=True)

        if not staging_candidates:
            logger.warning(f"Registry: no staging version available for rollback of '{model_name}'.")
            return None

        prev = staging_candidates[0]
        current_prod["status"] = "staging"
        prev["status"] = "production"
        _write_registry(registry)

    logger.info(
        f"Registry: rolled back {model_name}: {current_prod['version']} → staging, "
        f"{prev['version']} → production"
    )
    return prev


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the current production entry for a model, or None."""
    registry = _read_registry()
    versions = registry.get(model_name, [])
    return next((e for e in versions if e["status"] == "production"), None)


def list_versions(model_name: str) -> List[Dict[str, Any]]:
    """Return all versions of a model, newest first."""
    registry = _read_registry()
    versions = registry.get(model_name, [])
    return sorted(versions, key=lambda e: e["registered_at"], reverse=True)


def list_all_models() -> Dict[str, List[Dict[str, Any]]]:
    """Return the entire registry."""
    return _read_registry()


def add_tag(model_name: str, version: str, tag: str) -> Dict[str, Any]:
    """Add a tag string to a specific model version."""
    with _in_process_lock:
        registry = _read_registry()
        versions = registry.get(model_name, [])
        target = next((e for e in versions if e["version"] == version), None)
        if target is None:
            raise KeyError(f"Model '{model_name}' version '{version}' not found.")
        if tag not in target.get("tags", []):
            target.setdefault("tags", []).append(tag)
            _write_registry(registry)
    logger.info(f"Registry: added tag '{tag}' to {model_name}@{version}")
    return target


def compare_versions(
    model_name: str, version_a: str, version_b: str
) -> Dict[str, Any]:
    """
    Side-by-side metric comparison of two model versions.

    Returns a dict with keys: model_name, version_a, version_b, diff_table.
    diff_table is a list of {metric, value_a, value_b, delta} dicts.
    """
    registry = _read_registry()
    versions = registry.get(model_name, [])

    def _find(ver: str) -> Dict[str, Any]:
        found = next((e for e in versions if e["version"] == ver), None)
        if found is None:
            raise KeyError(f"Model '{model_name}' version '{ver}' not found.")
        return found

    entry_a = _find(version_a)
    entry_b = _find(version_b)

    metrics_a: Dict[str, Any] = entry_a.get("metrics", {})
    metrics_b: Dict[str, Any] = entry_b.get("metrics", {})
    all_metric_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

    diff_table = []
    for key in all_metric_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        try:
            delta = round(float(val_b) - float(val_a), 6) if val_a is not None and val_b is not None else None
        except (TypeError, ValueError):
            delta = None
        diff_table.append({"metric": key, version_a: val_a, version_b: val_b, "delta": delta})

    return {
        "model_name": model_name,
        "version_a": version_a,
        "version_b": version_b,
        "version_a_status": entry_a["status"],
        "version_b_status": entry_b["status"],
        "version_a_threshold": entry_a.get("optimal_threshold"),
        "version_b_threshold": entry_b.get("optimal_threshold"),
        "version_a_feature_version": entry_a.get("feature_version"),
        "version_b_feature_version": entry_b.get("feature_version"),
        "version_a_dataset_version": entry_a.get("dataset_version"),
        "version_b_dataset_version": entry_b.get("dataset_version"),
        "diff_table": diff_table,
    }
