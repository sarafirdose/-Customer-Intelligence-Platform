# Disaster Recovery Plan — Subscriber Intelligence Platform

## Overview

This document defines recovery procedures for data loss, model corruption, database failures, and complete system loss scenarios.

---

## Recovery Point Objective (RPO)

| Component | Target RPO |
|---|---|
| PostgreSQL database | 24 hours (daily backup) |
| Model artifacts | Immutable (re-train from data) |
| Model registry | Immediate (re-seed from artifacts) |
| Audit logs | Best effort (not replicated) |

## Recovery Time Objective (RTO)

| Scenario | Target RTO |
|---|---|
| API restart | < 2 minutes |
| Database restore | < 30 minutes |
| Full system rebuild | < 4 hours |
| Model retrain | < 2 hours (dataset dependent) |

---

## Scenario 1: API Process Crash

**Impact**: All predictions unavailable.

**Recovery Steps**:
1. Confirm process is dead: `Get-Process python` (Windows) or `ps aux | grep uvicorn`
2. Check last error: `tail -30 logs/error.log`
3. Restart: `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`
4. Verify: `curl http://localhost:8000/api/v1/ready`
5. Log incident in audit: `python -c "from backend.core.audit import log_audit_event; log_audit_event('SYSTEM', endpoint='disaster_recovery', result_summary={'incident': 'api_crash', 'resolved': True})"`

---

## Scenario 2: Database Failure

**Impact**: Customer data inaccessible. Batch scoring unavailable.

**Recovery Steps**:
1. Confirm DB status:
   ```bash
   pg_isready -h localhost -p 5432
   ```
2. Check PostgreSQL logs for cause (disk full, OOM, corruption).
3. Restore from backup:
   ```bash
   pg_restore -h localhost -U cip_user -d cip_db backup/cip_db_YYYY-MM-DD.dump
   ```
4. Run data integrity check:
   ```bash
   python -c "from backend.database.database import test_db_connection; print(test_db_connection())"
   ```
5. Restart API and verify health endpoint.

**Backup Schedule** (to be configured):
```bash
# Example daily backup cron
pg_dump -h localhost -U cip_user cip_db > backup/cip_db_$(date +%Y-%m-%d).dump
```

---

## Scenario 3: Model Artifact Corruption

**Impact**: Predictions fail with deserialization errors.

**Recovery Steps**:
1. Identify which artifact is corrupted:
   ```bash
   python -c "import pickle; pickle.load(open('artifacts/models/best_model.pkl','rb'))"
   ```
2. Check model registry for artifact path:
   ```bash
   python -c "from backend.ml.registry import get_production_model; import json; print(json.dumps(get_production_model('churn'), indent=2))"
   ```
3. Restore from version control or model store backup.
4. If backup unavailable, retrain:
   ```bash
   python scripts/train_all.py
   ```
5. Re-seed registry after retraining:
   ```bash
   python scripts/seed_registry.py
   ```

---

## Scenario 4: Model Registry Corruption

**Impact**: Registry API returns errors. Observability blind.

**Recovery Steps**:
1. Delete corrupted registry file and lock:
   ```bash
   rm artifacts/registry/model_registry.json
   rm -f artifacts/registry/model_registry.lock
   ```
2. Re-seed from existing artifacts:
   ```bash
   python scripts/seed_registry.py
   ```
3. Verify:
   ```bash
   curl http://localhost:8000/api/v1/observability/registry
   ```

---

## Scenario 5: Complete System Loss

**Impact**: All services unavailable. Full rebuild required.

**Recovery Steps**:
1. Restore source code from Git:
   ```bash
   git clone <repository-url>
   cd Customer-Intelligence-Platform
   ```
2. Restore environment:
   ```bash
   python -m venv venv
   venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   pip install -e .
   ```
3. Restore database from backup (see Scenario 2).
4. Restore model artifacts (from artifact store or retrain).
5. Populate `.env` from `.env.example`.
6. Run migrations:
   ```bash
   alembic upgrade head
   ```
7. Seed registry:
   ```bash
   python scripts/seed_registry.py
   ```
8. Start services and validate:
   ```bash
   python scripts/validate_enterprise.py
   ```

---

## Backup Checklist

| Asset | Backup Location | Frequency |
|---|---|---|
| PostgreSQL database | `backup/cip_db_YYYY-MM-DD.dump` | Daily |
| Model artifacts | `artifacts/` | After every training run |
| Model registry JSON | `artifacts/registry/` | After every registration |
| `.env` file | Secrets manager / encrypted vault | On change |
| Application logs | External log aggregator | Continuous |

---

*Reviewed and tested as part of Phase 7 Enterprise MLOps.*
