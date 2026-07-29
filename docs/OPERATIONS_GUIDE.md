# Operations Guide — Subscriber Intelligence Platform

## Overview

This guide covers day-to-day operational procedures for the Subscriber Intelligence Platform (CIP). It is intended for platform operators, MLOps engineers, and DevOps personnel.

---

## Quick Reference

| Task | Command |
|---|---|
| Start API (dev) | `uvicorn backend.api.main:app --reload --port 8000` |
| Start API (prod) | `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4` |
| Start Dashboard | `streamlit run dashboard/app.py` |
| Run all tests | `pytest tests/ -v` |
| Seed model registry | `python scripts/seed_registry.py` |
| Run drift check | `python scripts/run_drift_check.py --input reports/customer_intelligence.csv` |
| Run benchmarks | `python scripts/performance_test.py` |
| Run validation | `python scripts/validate_enterprise.py` |

---

## Daily Operations

### 1. Morning Health Check

Verify that all critical services are responding:

```bash
# API health
curl http://localhost:8000/api/v1/health

# Liveness probe
curl http://localhost:8000/api/v1/ready

# Metrics snapshot
curl http://localhost:8000/api/v1/metrics
```

Expected: all return HTTP 200 with `"status": "healthy"` or `"status": "ready"`.

### 2. Review Daily Metrics

Metrics are flushed to `logs/metrics_YYYY-MM-DD.json` automatically at 00:05 UTC by the scheduler.

Manual flush (if needed):
```python
from backend.core.metrics import metrics
metrics.flush_daily_summary()
```

### 3. Check Audit Log

```bash
# View last 20 audit events
tail -20 logs/audit.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
```

### 4. Scheduler Status

Via API:
```bash
curl http://localhost:8000/api/v1/observability/scheduler/jobs
```

Via Operations Dashboard (Streamlit page 9).

---

## Weekly Operations

### 1. Drift Report Review

The scheduler runs drift detection every Monday at 01:00 UTC automatically.

Manual run:
```bash
python scripts/run_drift_check.py --input reports/customer_intelligence.csv
```

Review: `reports/feature_drift_report.md` and `reports/drift_summary.json`.

**Severity Actions:**

| Severity | Action |
|---|---|
| ✅ Normal | No action required |
| ⚠️ Warning | Increase monitoring frequency, investigate data source changes |
| 🚨 Critical | Initiate emergency retraining review, escalate to ML team |

### 2. Scheduler History Review

```bash
# View last 10 job executions
tail -10 logs/scheduler_history.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
```

---

## Monthly Operations

### 1. Model Registry Review

```bash
curl http://localhost:8000/api/v1/observability/registry
```

Check that all three models (`churn`, `ltv`, `segmentation`) have a production version.

### 2. Performance Benchmark

```bash
python scripts/performance_test.py
```

Compare `docs/performance/YYYY-MM-DD.md` to previous months to detect regressions.

### 3. Model Evaluation

The scheduler triggers `monthly_model_evaluation` on the 1st of each month at 02:00 UTC. This logs current production model metadata to the application log.

---

## Log File Reference

| File | Description | Rotation |
|---|---|---|
| `logs/app.log` | General application log | 10 MB, 5 backups |
| `logs/error.log` | ERROR+ only | 5 MB, 3 backups |
| `logs/audit.jsonl` | Structured audit trail (JSONL) | Manual / external |
| `logs/scheduler_history.jsonl` | Job execution history | Manual cleanup (30-day auto) |
| `logs/metrics_YYYY-MM-DD.json` | Daily metrics snapshots | Auto-deleted after 30 days |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ENV` | `development` | Environment name |
| `DEBUG` | `true` | Enable debug mode |
| `LOG_JSON` | `false` | JSON log format (enable in production) |
| `METRICS_ENABLED` | `true` | Enable metrics collection |
| `SCHEDULER_ENABLED` | `true` | Enable background scheduler |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-IP rate limit |
| `DRIFT_WARNING_THRESHOLD` | `0.10` | PSI warning boundary |
| `DRIFT_CRITICAL_THRESHOLD` | `0.25` | PSI critical boundary |
| `REGISTRY_PATH` | `artifacts/registry/model_registry.json` | Registry file location |

---

## Contacts and Escalation

| Issue | Owner |
|---|---|
| API down | Platform Engineering |
| Drift Critical | ML Engineering |
| DB connection failed | Database Operations |
| Model performance degraded | Data Science |
