# Runbook — Subscriber Intelligence Platform

> [!IMPORTANT]
> This runbook covers emergency response procedures. Keep it accessible offline.

---

## Incident: API Server Unresponsive

**Symptoms**: `/api/v1/health` returns timeout or connection refused.

**Steps**:
1. Check if uvicorn process is running:
   ```bash
   # On Linux/Mac
   ps aux | grep uvicorn
   # On Windows
   Get-Process python
   ```
2. Check application log for startup errors:
   ```bash
   tail -50 logs/error.log
   ```
3. Verify PostgreSQL is reachable:
   ```bash
   curl http://localhost:8000/api/v1/health
   # If DB is the issue: status=degraded, database=disconnected
   ```
4. Restart the API:
   ```bash
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```
5. If persists: check `logs/app.log` for import errors or model loading failures.

---

## Incident: Database Connection Failed

**Symptoms**: `/health` returns `"database": "disconnected"`.

**Steps**:
1. Verify PostgreSQL is running:
   ```bash
   # Docker
   docker compose ps
   # Local
   pg_isready -h localhost -p 5432
   ```
2. Check credentials in `.env`:
   ```
   DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
   ```
3. Test direct connection:
   ```bash
   psql -h localhost -U cip_user -d cip_db
   ```
4. Check PostgreSQL logs for `max_connections` or disk space issues.
5. Restart PostgreSQL if necessary.

---

## Incident: Critical Drift Alert

**Symptoms**: `reports/drift_summary.json` shows `"overall_severity": "Critical"`.

**Steps**:
1. Review which features are drifting:
   ```bash
   cat reports/feature_drift_report.md
   ```
2. Check PSI values against thresholds:
   - `monthly_charges` or `tenure_months` critical → data pipeline issue
   - `contract_type` critical → business/market shift
3. Identify root cause:
   - ETL pipeline date range error?
   - Data source change?
   - Genuine market shift?
4. Escalate to ML team with the drift report.
5. Decide: retrain, accept, or investigate further.
6. Document finding in `logs/audit.jsonl` via audit module.

---

## Incident: Scheduler Not Running

**Symptoms**: Jobs not appearing in `GET /api/v1/observability/scheduler/jobs`.

**Steps**:
1. Check `SCHEDULER_ENABLED=true` in `.env`.
2. Look for APScheduler errors in `logs/app.log`:
   ```bash
   grep "Scheduler" logs/app.log | tail -20
   ```
3. Check `logs/scheduler_history.jsonl` for recent job history.
4. Restart the API server (scheduler starts in FastAPI lifespan).

---

## Incident: Model Registry Corruption

**Symptoms**: `GET /api/v1/observability/registry` returns empty or malformed JSON.

**Steps**:
1. Check if registry file exists:
   ```bash
   ls artifacts/registry/model_registry.json
   ```
2. Validate JSON:
   ```bash
   python -c "import json; json.load(open('artifacts/registry/model_registry.json'))"
   ```
3. If corrupted, restore from backup:
   ```bash
   # The registry lock file (.lock) should be absent for fresh write
   rm -f artifacts/registry/model_registry.lock
   cp artifacts/registry/model_registry.json.bak artifacts/registry/model_registry.json
   ```
4. If no backup, re-seed:
   ```bash
   python scripts/seed_registry.py
   ```

---

## Incident: Prediction Latency Spike

**Symptoms**: `GET /api/v1/metrics` shows `p95_ms > 500`.

**Steps**:
1. Check CPU and memory in `/metrics`:
   ```json
   {"cpu_percent": 95, "memory_mb": 3800}
   ```
2. If CPU is pegged: check for batch jobs running concurrently.
3. Check if model loaded correctly:
   ```bash
   python -c "import pickle; pickle.load(open('artifacts/models/best_model.pkl','rb'))"
   ```
4. Review `logs/error.log` for prediction exceptions.
5. If memory is near limit, restart uvicorn workers.

---

## Rollback Procedure

If a new model version causes problems:

```bash
# Via Python CLI
python -c "
from backend.ml.registry import rollback
result = rollback('churn')
print(f'Rolled back to: {result[\"version\"]}')
"
```

Or via API:
```bash
# Get current production
curl http://localhost:8000/api/v1/observability/registry/churn/production

# Promote previous staging version
# (use the promote endpoint when implemented in admin UI)
```

---

## Emergency Contacts

| Severity | Escalation Path |
|---|---|
| P0 — Total outage | On-call Engineer → Engineering Lead |
| P1 — Degraded predictions | ML Engineer → Data Science Lead |
| P2 — Drift Critical | ML Engineer (business hours) |
| P3 — Performance issues | Platform Engineer (next business day) |
