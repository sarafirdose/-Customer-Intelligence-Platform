# Model Registry — Subscriber Intelligence Platform

## Overview

The Model Registry is a local file-based system that tracks all trained model versions, supports status promotions, rollbacks, tagging, and metric comparisons. It is backed by `artifacts/registry/model_registry.json` with `FileLock` for concurrent-safe writes.

---

## Registry Schema

Each entry in the registry contains:

```json
{
  "model_name": "churn",
  "version": "v1.0.0",
  "model_type": "classifier",
  "artifact_path": "artifacts/models/best_model.pkl",
  "metrics": {
    "roc_auc": 0.847,
    "f1": 0.633,
    "accuracy": 0.781
  },
  "optimal_threshold": 0.61,
  "dataset_rows": 7043,
  "dataset_version": "v1.0",
  "feature_version": "v1.0",
  "git_commit": "e4b855c",
  "python_version": "3.12.0",
  "status": "production",
  "tags": ["stable", "approved", "july-release"],
  "notes": "Initial production model.",
  "registered_at": "2026-07-27T07:55:12Z"
}
```

---

## Status Lifecycle

```
development  →  staging  →  production
                             ↓
                           archived
```

| Status | Description |
|---|---|
| `development` | Freshly trained, not yet validated |
| `staging` | Validated, ready for production review |
| `production` | Active model serving predictions |
| `archived` | Retired, kept for historical reference |

> [!IMPORTANT]
> Only **one version per model name** can hold `production` status at any time. Promoting a new version to `production` automatically demotes the current production version to `staging`.

---

## Supported Models

| Model Name | Type | Description |
|---|---|---|
| `churn` | classifier | Binary churn prediction |
| `ltv` | regressor | Subscriber Lifetime Value |
| `segmentation` | clustering | K-Means customer segments |

---

## Python API

### Register a model

```python
from backend.ml.registry import register_model

register_model(
    model_name="churn",
    version="v1.1.0",
    model_type="classifier",
    artifact_path="artifacts/models/best_model.pkl",
    metrics={"roc_auc": 0.856, "f1": 0.641},
    dataset_rows=7043,
    git_commit="abc123",
    status="development",
    tags=["retrained", "august-release"],
    notes="Retrained with updated features.",
)
```

### Promote to production

```python
from backend.ml.registry import promote

promote("churn", "v1.1.0", "production")
# Automatically demotes v1.0.0 → staging
```

### Rollback

```python
from backend.ml.registry import rollback

result = rollback("churn")
print(f"Rolled back to: {result['version']}")
```

### Compare versions

```python
from backend.ml.registry import compare_versions

result = compare_versions("churn", "v1.0.0", "v1.1.0")
for row in result["diff_table"]:
    print(row["metric"], row["v1.0.0"], row["v1.1.0"], row["delta"])
```

### Add a tag

```python
from backend.ml.registry import add_tag

add_tag("churn", "v1.1.0", "approved")
```

---

## REST API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/observability/registry` | Full registry |
| GET | `/api/v1/observability/registry/{model}` | All versions |
| GET | `/api/v1/observability/registry/{model}/production` | Active production |
| GET | `/api/v1/observability/registry/{model}/compare?v1=v1.0.0&v2=v1.1.0` | Metric diff |

---

## File Location

```
artifacts/
  registry/
    model_registry.json   ← Registry data
    model_registry.lock   ← FileLock (auto-created, do not edit)
```

---

## Seeding

Populate the registry from existing trained artifacts:

```bash
python scripts/seed_registry.py
```

---

## Concurrency Safety

All write operations use a `FileLock` (from the `filelock` library) + an in-process `threading.RLock`. This prevents corruption when:
- Multiple API workers register models simultaneously
- A scheduler job updates the registry while a request is in progress

---

*Part of Phase 7 Enterprise MLOps.*
