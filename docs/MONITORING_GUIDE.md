# Monitoring Guide — Subscriber Intelligence Platform

## Overview

This guide describes the monitoring architecture, metrics definitions, alert thresholds, and operational dashboards for the Subscriber Intelligence Platform.

---

## Monitoring Stack

| Component | Tool | Location |
|---|---|---|
| API metrics | Custom MetricsCollector | `/api/v1/metrics` |
| Health checks | FastAPI endpoints | `/api/v1/health`, `/api/v1/ready` |
| Drift detection | PSI + Chi-squared | `reports/drift/` |
| Audit trail | JSONL audit log | `logs/audit.jsonl` |
| Scheduler history | JSONL log | `logs/scheduler_history.jsonl` |
| Ops Dashboard | Streamlit page 9 | `dashboard/pages/9_Operations.py` |

---

## API Metrics Reference

All metrics exposed at `GET /api/v1/metrics`.

### Request Metrics

| Metric | Description | Alert Threshold |
|---|---|---|
| `requests.total` | Total API requests since startup | N/A |
| `requests.errors` | Total 5xx error responses | — |
| `requests.error_rate` | Fraction of requests that errored | > 0.05 (5%) |
| `requests.per_endpoint` | Request count per path | — |

### Latency Metrics

| Metric | Description | Alert Threshold |
|---|---|---|
| `latency_ms.avg` | Average response time | > 200 ms |
| `latency_ms.p50` | Median response time | > 100 ms |
| `latency_ms.p95` | 95th percentile latency | > 500 ms |
| `latency_ms.p99` | 99th percentile latency | > 1000 ms |

### Prediction Metrics

| Metric | Description |
|---|---|
| `predictions.count` | Total predictions served |
| `predictions.avg_churn_probability` | Rolling average churn score |
| `predictions.avg_ltv` | Rolling average projected LTV |

### System Metrics

| Metric | Description | Alert Threshold |
|---|---|---|
| `cpu_percent` | Process CPU usage | > 85% |
| `memory_mb` | Process RSS memory | > 2000 MB |
| `uptime_seconds` | Seconds since startup | — |

---

## Health Check Reference

### `GET /api/v1/health`

Deep health check. Validates:
- PostgreSQL connectivity (live query)
- Required model artifact presence

**Response fields**:
```json
{
  "status": "healthy | degraded",
  "database": "connected | disconnected",
  "artifacts": "present | missing: [...]",
  "api": "running",
  "version": "1.0.0",
  "environment": "production"
}
```

### `GET /api/v1/ready`

Lightweight liveness. No I/O. Always returns 200 if process is alive.

```json
{"status": "ready", "version": "1.0.0"}
```

---

## Drift Detection Reference

### Numerical Features — PSI (Population Stability Index)

| Feature | Baseline Source |
|---|---|
| `tenure_months` | `artifacts/baseline/mean.json` |
| `monthly_charges` | `artifacts/baseline/mean.json` |
| `total_charges` | `artifacts/baseline/mean.json` |
| `charges_ratio` | `artifacts/baseline/mean.json` |
| `total_services` | `artifacts/baseline/mean.json` |
| `total_charges_log` | `artifacts/baseline/mean.json` |

**PSI Severity**:

| PSI Range | Severity | Action |
|---|---|---|
| < 0.10 | ✅ Normal | No action |
| 0.10 – 0.25 | ⚠️ Warning | Monitor closely, investigate data source |
| >= 0.25 | 🚨 Critical | Escalate to ML team, consider retraining |

### Categorical Features — Proportional Shift

| Feature | Baseline Source |
|---|---|
| `contract_type` | `artifacts/baseline/category_distribution.json` |
| `payment_method` | `artifacts/baseline/category_distribution.json` |
| `internet_service` | `artifacts/baseline/category_distribution.json` |
| `tenure_group` | `artifacts/baseline/category_distribution.json` |

Max shift thresholds match PSI thresholds.

### Drift History

All drift reports saved to `reports/drift/YYYY-MM-DD.json`.
Latest summary always at `reports/drift_summary.json`.
Human-readable at `reports/feature_drift_report.md`.

---

## Scheduler Monitoring

### Scheduled Jobs

| Job | Schedule | Log |
|---|---|---|
| `daily_metrics_flush` | 00:05 UTC daily | `logs/scheduler_history.jsonl` |
| `weekly_drift_report` | Monday 01:00 UTC | `logs/scheduler_history.jsonl` |
| `monthly_model_evaluation` | 1st of month 02:00 UTC | `logs/scheduler_history.jsonl` |
| `log_rotation_cleanup` | 00:10 UTC daily | `logs/scheduler_history.jsonl` |

### Job History Format

```json
{
  "job": "weekly_drift_report",
  "status": "success",
  "start_time": "2026-07-28T01:00:00Z",
  "end_time": "2026-07-28T01:00:03Z",
  "duration_seconds": 3.21,
  "error": null
}
```

---

## Audit Log Reference

All events written to `logs/audit.jsonl`.

### Event Types

| Type | Description |
|---|---|
| `PREDICTION` | Single customer prediction request |
| `BATCH_JOB` | Batch scoring job completed |
| `EXPORT` | Data export action |
| `ERROR` | Application error |
| `USER_ACTION` | User-triggered operation |
| `SYSTEM` | System-level event (startup, health checks) |
| `REGISTRY` | Model registry operation |
| `DRIFT_CHECK` | Drift analysis execution |
| `SCHEDULER` | Scheduler job execution |

### Searching Audit Logs

```bash
# Find all PREDICTION events for a customer
grep PREDICTION logs/audit.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l))) for l in sys.stdin if 'C001' in l]"

# Find all errors in last 100 events
tail -100 logs/audit.jsonl | grep ERROR

# Count by event type
python -c "
import json
from collections import Counter
with open('logs/audit.jsonl') as f:
    events = [json.loads(l) for l in f if l.strip()]
counts = Counter(e['event_type'] for e in events)
print(dict(counts))
"
```

---

## Operations Dashboard

Access via Streamlit dashboard → Page 9 (Operations):

```bash
streamlit run dashboard/app.py
```

Navigate to: **⚙️ Enterprise Operations Dashboard**

Provides live views of:
- System health strip
- CPU / memory / latency / error rate KPIs
- Model registry table
- PSI drift trend charts
- Scheduler job status
- Recent audit log entries (last 50)

---

*Part of Phase 7 Enterprise MLOps.*
