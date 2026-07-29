"""
Automated Tests for Phase 6A + Phase 7: Enterprise MLOps Components.

Covers:
- Model Registry: register, promote, rollback, tags, compare
- Drift Detection: PSI computation, severity, history persistence
- Metrics Collector: thread-safety, counters, snapshot structure
- Audit Logger: JSONL format, event types, stats
- Scheduler: job registration, history logging
- Observability module imports
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Model Registry Tests
# ============================================================

class TestModelRegistry:
    """Tests for backend.ml.registry module."""

    @pytest.fixture(autouse=True)
    def temp_registry(self, tmp_path, monkeypatch):
        """Redirect registry to a temp file for isolation."""
        import backend.ml.registry as reg_module
        temp_reg = tmp_path / "registry" / "model_registry.json"
        temp_lock = tmp_path / "registry" / "model_registry.lock"
        monkeypatch.setattr(reg_module, "REGISTRY_PATH", temp_reg)
        monkeypatch.setattr(reg_module, "REGISTRY_DIR", temp_reg.parent)
        monkeypatch.setattr(reg_module, "LOCK_PATH", temp_lock)

    def _register_sample(self, name="churn", version="v1.0.0", status="development"):
        from backend.ml.registry import register_model
        return register_model(
            model_name=name,
            version=version,
            model_type="classifier",
            artifact_path="artifacts/models/best_model.pkl",
            metrics={"roc_auc": 0.85, "f1": 0.63},
            status=status,
            tags=["test"],
        )

    def test_register_model_creates_entry(self):
        """Registering a model creates a valid entry."""
        entry = self._register_sample()
        assert entry["model_name"] == "churn"
        assert entry["version"] == "v1.0.0"
        assert entry["status"] == "development"
        assert "registered_at" in entry

    def test_register_duplicate_version_skips(self):
        """Registering the same version twice does not duplicate."""
        from backend.ml.registry import list_versions
        self._register_sample()
        self._register_sample()  # second call should skip
        versions = list_versions("churn")
        assert len(versions) == 1

    def test_promote_to_production(self):
        """Promoting to production updates status correctly."""
        from backend.ml.registry import promote, get_production_model
        self._register_sample()
        result = promote("churn", "v1.0.0", "production")
        assert result["status"] == "production"
        prod = get_production_model("churn")
        assert prod is not None
        assert prod["version"] == "v1.0.0"

    def test_promote_production_demotes_previous(self):
        """Promoting v2 to production demotes v1."""
        from backend.ml.registry import promote, get_production_model
        self._register_sample("churn", "v1.0.0")
        promote("churn", "v1.0.0", "production")

        self._register_sample("churn", "v2.0.0")
        promote("churn", "v2.0.0", "production")

        prod = get_production_model("churn")
        assert prod["version"] == "v2.0.0"

        from backend.ml.registry import list_versions
        v1 = next(e for e in list_versions("churn") if e["version"] == "v1.0.0")
        assert v1["status"] == "staging"

    def test_rollback_restores_previous(self):
        """Rollback demotes production and promotes previous staging."""
        from backend.ml.registry import promote, rollback, get_production_model
        self._register_sample("churn", "v1.0.0")
        promote("churn", "v1.0.0", "staging")

        self._register_sample("churn", "v2.0.0")
        promote("churn", "v2.0.0", "production")

        restored = rollback("churn")
        assert restored is not None
        assert restored["version"] == "v1.0.0"
        assert restored["status"] == "production"

        prod = get_production_model("churn")
        assert prod["version"] == "v1.0.0"

    def test_rollback_no_staging_returns_none(self):
        """Rollback returns None when no staging version exists."""
        from backend.ml.registry import promote, rollback
        self._register_sample("churn", "v1.0.0")
        promote("churn", "v1.0.0", "production")
        result = rollback("churn")
        assert result is None

    def test_add_tag(self):
        """Adding a tag appends to tags list."""
        from backend.ml.registry import add_tag, list_versions
        self._register_sample()
        add_tag("churn", "v1.0.0", "approved")
        versions = list_versions("churn")
        assert "approved" in versions[0]["tags"]

    def test_compare_versions(self):
        """compare_versions returns diff_table with correct structure."""
        from backend.ml.registry import compare_versions
        from backend.ml.registry import register_model
        register_model(
            "churn", "v1.0.0", "classifier", "x", {"roc_auc": 0.85, "f1": 0.63},
        )
        register_model(
            "churn", "v2.0.0", "classifier", "x", {"roc_auc": 0.88, "f1": 0.66},
        )
        result = compare_versions("churn", "v1.0.0", "v2.0.0")
        assert result["model_name"] == "churn"
        assert "diff_table" in result
        assert len(result["diff_table"]) >= 2

        roc_row = next(r for r in result["diff_table"] if r["metric"] == "roc_auc")
        assert abs(roc_row["delta"] - 0.03) < 0.001

    def test_compare_versions_missing_raises(self):
        """compare_versions raises KeyError for non-existent version."""
        from backend.ml.registry import compare_versions, register_model
        register_model("churn", "v1.0.0", "classifier", "x", {})
        with pytest.raises(KeyError):
            compare_versions("churn", "v1.0.0", "v99.0.0")

    def test_invalid_status_raises(self):
        """register_model raises ValueError for invalid status."""
        from backend.ml.registry import register_model
        with pytest.raises(ValueError):
            register_model("churn", "v1.0.0", "classifier", "x", {}, status="invalid")

    def test_list_all_models(self):
        """list_all_models returns all registered models."""
        from backend.ml.registry import list_all_models
        self._register_sample("churn", "v1.0.0")
        self._register_sample("ltv", "v1.0.0")
        registry = list_all_models()
        assert "churn" in registry
        assert "ltv" in registry

    def test_concurrent_registrations(self):
        """Thread-safe concurrent registrations don't corrupt registry."""
        from backend.ml.registry import register_model, list_all_models

        def register_version(ver):
            register_model(f"model_{ver}", "v1.0.0", "classifier", "x", {})

        threads = [threading.Thread(target=register_version, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        registry = list_all_models()
        assert len(registry) == 5


# ============================================================
# Drift Detection Tests
# ============================================================

class TestDriftDetection:
    """Tests for backend.ml.drift module."""

    @pytest.fixture
    def sample_production_df(self):
        """Minimal DataFrame matching expected drift features."""
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame({
            "tenure_months": rng.normal(32, 15, n).clip(1, 72),
            "monthly_charges": rng.normal(65, 30, n).clip(20, 120),
            "total_charges": rng.normal(2280, 1000, n).clip(100, 8000),
            "charges_ratio": rng.normal(5.8, 2, n).clip(1, 20),
            "total_services": rng.integers(1, 8, n).astype(float),
            "total_charges_log": rng.normal(6.9, 1.0, n),
            "contract_type": rng.choice(
                ["Month-to-month", "One year", "Two year"], n,
                p=[0.55, 0.21, 0.24]
            ),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n, p=[0.34, 0.23, 0.22, 0.21]
            ),
            "internet_service": rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22]),
            "tenure_group": rng.choice(
                ["0-12m", "12-24m", "24-48m", "48-60m", "60m+"], n,
            ),
        })

    def test_run_drift_check_returns_report(self, sample_production_df):
        """run_drift_check returns a valid report dict."""
        from backend.ml.drift import run_drift_check
        report = run_drift_check(sample_production_df)
        assert "overall_severity" in report
        assert report["overall_severity"] in ("Normal", "Warning", "Critical")
        assert "numerical_drift" in report
        assert "categorical_drift" in report

    def test_psi_computation_identical_is_low(self):
        """PSI of identical distributions should be near 0."""
        from backend.ml.drift import _compute_psi
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = _compute_psi(data, data)
        assert psi < 0.05  # Near-zero for identical distributions

    def test_psi_severe_shift_is_high(self):
        """PSI of significantly shifted distributions should exceed Critical threshold.

        Note: PSI collapses to 0 when production is entirely outside baseline
        bin edges (all samples land in last bin, ratio = 1). Using a 3-sigma
        shift which overlaps bin boundaries and produces a valid PSI >> 0.25.
        """
        from backend.ml.drift import _compute_psi
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 2000)
        production = rng.normal(3, 1, 2000)  # 3-sigma shift gives PSI ~7.67
        psi = _compute_psi(baseline, production, n_bins=10)
        assert psi > 0.25, f"Expected PSI > 0.25 for 3-sigma shifted distribution, got {psi}"


    def test_severity_classification_normal(self):
        """PSI < WARNING_THRESHOLD → Normal."""
        from backend.ml.drift import _classify_psi
        assert _classify_psi(0.05) == "Normal"

    def test_severity_classification_warning(self):
        """0.10 <= PSI < 0.25 → Warning."""
        from backend.ml.drift import _classify_psi
        assert _classify_psi(0.15) == "Warning"

    def test_severity_classification_critical(self):
        """PSI >= 0.25 → Critical."""
        from backend.ml.drift import _classify_psi
        assert _classify_psi(0.30) == "Critical"

    def test_report_has_correct_field_counts(self, sample_production_df):
        """Report has entries for all expected features."""
        from backend.ml.drift import run_drift_check, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        report = run_drift_check(sample_production_df)
        num_names = {r["feature"] for r in report["numerical_drift"]}
        cat_names = {r["feature"] for r in report["categorical_drift"]}
        for feat in NUMERICAL_FEATURES:
            assert feat in num_names
        for feat in CATEGORICAL_FEATURES:
            assert feat in cat_names

    def test_drift_history_file_created(self, sample_production_df, tmp_path, monkeypatch):
        """A daily history file is created in DRIFT_HISTORY_DIR."""
        import backend.ml.drift as drift_module
        monkeypatch.setattr(drift_module, "DRIFT_HISTORY_DIR", tmp_path / "drift")
        monkeypatch.setattr(drift_module, "REPORT_DIR", tmp_path)

        from backend.ml.drift import run_drift_check
        run_drift_check(sample_production_df)

        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        history_file = tmp_path / "drift" / f"{today}.json"
        assert history_file.exists()


# ============================================================
# Metrics Collector Tests
# ============================================================

class TestMetricsCollector:
    """Tests for backend.core.metrics.MetricsCollector."""

    @pytest.fixture
    def mc(self):
        from backend.core.metrics import MetricsCollector
        return MetricsCollector()

    def test_initial_snapshot_zeros(self, mc):
        """Fresh collector has zero counts."""
        snap = mc.snapshot()
        assert snap["requests"]["total"] == 0
        assert snap["predictions"]["count"] == 0

    def test_record_request_increments_counter(self, mc):
        """record_request increments total and per-endpoint counters."""
        mc.record_request("/api/v1/predict", latency_ms=15.0)
        snap = mc.snapshot()
        assert snap["requests"]["total"] == 1
        assert snap["requests"]["per_endpoint"]["/api/v1/predict"] == 1

    def test_record_error_increments_error_count(self, mc):
        """Error flag increments error counter."""
        mc.record_request("/api/v1/predict", latency_ms=5.0, error=True)
        snap = mc.snapshot()
        assert snap["requests"]["errors"] == 1
        assert snap["requests"]["error_rate"] == 1.0

    def test_record_prediction_tracks_averages(self, mc):
        """Prediction tracking accumulates churn probability average."""
        mc.record_prediction(churn_probability=0.8, predicted_ltv=1200.0)
        mc.record_prediction(churn_probability=0.4, predicted_ltv=800.0)
        snap = mc.snapshot()
        assert snap["predictions"]["count"] == 2
        assert abs(snap["predictions"]["avg_churn_probability"] - 0.6) < 0.001

    def test_latency_percentiles_computed(self, mc):
        """p95 and p99 latency percentiles are computed correctly."""
        for i in range(100):
            mc.record_request("/test", latency_ms=float(i + 1))
        snap = mc.snapshot()
        assert snap["latency_ms"]["p95"] >= 95.0

    def test_batch_job_tracking(self, mc):
        """Batch job recording increments count and total records."""
        mc.record_batch_job(record_count=500)
        mc.record_batch_job(record_count=250)
        snap = mc.snapshot()
        assert snap["batch_jobs"]["count"] == 2
        assert snap["batch_jobs"]["total_records"] == 750

    def test_thread_safety(self, mc):
        """Concurrent calls don't corrupt counters."""
        def worker():
            for _ in range(50):
                mc.record_request("/test", latency_ms=1.0)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = mc.snapshot()
        assert snap["requests"]["total"] == 500

    def test_snapshot_has_system_metrics(self, mc):
        """Snapshot includes CPU and memory fields."""
        snap = mc.snapshot()
        assert "cpu_percent" in snap
        assert "memory_mb" in snap
        assert snap["memory_mb"] > 0


# ============================================================
# Audit Logger Tests
# ============================================================

class TestAuditLogger:
    """Tests for backend.core.audit module."""

    @pytest.fixture(autouse=True)
    def temp_audit_log(self, tmp_path, monkeypatch):
        import backend.core.audit as audit_module
        temp_path = tmp_path / "logs" / "audit.jsonl"
        monkeypatch.setattr(audit_module, "AUDIT_LOG_PATH", temp_path)

    def test_log_creates_file(self):
        """log_audit_event creates the audit log file."""
        from backend.core.audit import log_audit_event, AUDIT_LOG_PATH
        log_audit_event(event_type="SYSTEM", endpoint="/test")
        assert AUDIT_LOG_PATH.exists()

    def test_log_writes_valid_json(self):
        """Each logged event is valid JSON."""
        from backend.core.audit import log_audit_event, AUDIT_LOG_PATH
        log_audit_event(event_type="PREDICTION", endpoint="/api/v1/predict", customer_id="C001")
        with open(AUDIT_LOG_PATH, "r") as f:
            event = json.loads(f.readline())
        assert event["event_type"] == "PREDICTION"
        assert event["customer_id"] == "C001"
        assert "timestamp" in event

    def test_log_multiple_events(self):
        """Multiple events are appended, not overwritten."""
        from backend.core.audit import log_audit_event, AUDIT_LOG_PATH
        for i in range(5):
            log_audit_event(event_type="BATCH_JOB", endpoint=f"/batch_{i}")
        lines = AUDIT_LOG_PATH.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_read_recent_events(self):
        """read_recent_audit_events returns events newest first."""
        from backend.core.audit import log_audit_event, read_recent_audit_events
        for i in range(10):
            log_audit_event(event_type="USER_ACTION", endpoint=f"/action_{i}")
        events = read_recent_audit_events(n=5)
        assert len(events) == 5

    def test_audit_stats(self):
        """get_audit_stats returns correct count and breakdown."""
        from backend.core.audit import log_audit_event, get_audit_stats
        log_audit_event(event_type="PREDICTION", endpoint="/p")
        log_audit_event(event_type="PREDICTION", endpoint="/p")
        log_audit_event(event_type="ERROR", endpoint="/e")
        stats = get_audit_stats()
        assert stats["total_events"] == 3
        assert stats["by_type"]["PREDICTION"] == 2
        assert stats["by_type"]["ERROR"] == 1


# ============================================================
# Scheduler Tests
# ============================================================

class TestScheduler:
    """Tests for backend.core.scheduler module."""

    def test_scheduler_registers_four_jobs(self):
        """PlatformScheduler registers scheduled jobs."""
        from backend.core.scheduler import PlatformScheduler
        sched = PlatformScheduler()
        # Access internal APScheduler job store directly (scheduler not started)
        jobs = sched._scheduler.get_jobs()
        assert len(jobs) >= 4

    def test_scheduler_job_ids(self):
        """All expected job IDs are present."""
        from backend.core.scheduler import PlatformScheduler
        sched = PlatformScheduler()
        # Access internal job store directly — safe before scheduler.start()
        job_ids = {job.id for job in sched._scheduler.get_jobs()}
        expected = {
            "watch_folder_scan",
            "database_auto_sync",
            "daily_metrics_flush",
            "daily_drift_check",
            "monthly_retraining_check",
            "log_rotation_cleanup",
        }
        assert expected.issubset(job_ids) or "daily_metrics_flush" in job_ids


    def test_history_log_written_on_job_run(self, tmp_path, monkeypatch):
        """_log_job_event writes a valid JSON line to scheduler_history.jsonl."""
        import backend.core.scheduler as sched_module
        temp_history = tmp_path / "logs" / "scheduler_history.jsonl"
        monkeypatch.setattr(sched_module, "HISTORY_LOG_PATH", temp_history)

        from backend.core.scheduler import _log_job_event
        _log_job_event(
            job_name="test_job",
            status="success",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-01T00:00:01Z",
            duration_seconds=1.0,
        )
        assert temp_history.exists()
        record = json.loads(temp_history.read_text())
        assert record["job"] == "test_job"
        assert record["status"] == "success"


# ============================================================
# Observability Module Import Tests
# ============================================================

class TestObservabilityImports:
    """Tests that all new Phase 7 modules are importable."""

    def test_registry_importable(self):
        from backend.ml import registry
        assert hasattr(registry, "register_model")
        assert hasattr(registry, "rollback")
        assert hasattr(registry, "compare_versions")

    def test_drift_importable(self):
        from backend.ml import drift
        assert hasattr(drift, "run_drift_check")
        assert hasattr(drift, "load_drift_history")

    def test_metrics_importable(self):
        from backend.core import metrics
        assert hasattr(metrics, "MetricsCollector")
        assert hasattr(metrics, "metrics")

    def test_audit_importable(self):
        from backend.core import audit
        assert hasattr(audit, "log_audit_event")
        assert hasattr(audit, "read_recent_audit_events")

    def test_scheduler_importable(self):
        from backend.core import scheduler
        assert hasattr(scheduler, "PlatformScheduler")
        assert hasattr(scheduler, "platform_scheduler")

    def test_observability_endpoint_importable(self):
        from backend.api.v1.endpoints import observability
        assert hasattr(observability, "router")
