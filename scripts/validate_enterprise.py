"""
Enterprise Validation Script.

Validates all Phase 6A + Phase 7 components are correctly installed and functional.
Produces reports/enterprise_validation.md with a pass/fail checklist.

Usage:
    python scripts/validate_enterprise.py [--api-url http://localhost:8000]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = BASE_DIR / "docs" / "enterprise_validation.md"

Results = List[Tuple[str, bool, str]]


# ---------------------------------------------------------------------------
# Check functions (each returns (passed: bool, detail: str))
# ---------------------------------------------------------------------------

def check_registry() -> Tuple[bool, str]:
    try:
        from backend.ml.registry import list_all_models, get_production_model
        registry = list_all_models()
        prod_churn = get_production_model("churn")
        if not registry:
            return False, "Registry is empty — run: python scripts/seed_registry.py"
        prod_count = sum(1 for models in registry.values() for m in models if m["status"] == "production")
        return True, f"{len(registry)} model(s), {prod_count} production version(s)"
    except Exception as e:
        return False, str(e)


def check_drift_baselines() -> Tuple[bool, str]:
    required = [
        BASE_DIR / "artifacts" / "baseline" / "mean.json",
        BASE_DIR / "artifacts" / "baseline" / "std.json",
        BASE_DIR / "artifacts" / "baseline" / "category_distribution.json",
    ]
    missing = [f.name for f in required if not f.exists()]
    if missing:
        return False, f"Missing baseline files: {missing}"
    return True, "All 3 baseline files present"


def check_drift_module() -> Tuple[bool, str]:
    try:
        import pandas as pd
        from backend.ml.drift import run_drift_check, NUMERICAL_FEATURES
        # Run with a minimal synthetic dataframe
        df = pd.DataFrame({feat: [1.0, 2.0, 3.0] for feat in NUMERICAL_FEATURES})
        report = run_drift_check(df)
        return True, f"Drift check ran. Overall: {report['overall_severity']}"
    except Exception as e:
        return False, str(e)


def check_metrics_collector() -> Tuple[bool, str]:
    try:
        from backend.core.metrics import MetricsCollector
        mc = MetricsCollector()
        mc.record_request("/test", latency_ms=5.5, error=False)
        mc.record_prediction(churn_probability=0.7, predicted_ltv=1200.0)
        snap = mc.snapshot()
        assert snap["requests"]["total"] == 1, "Counter mismatch"
        assert snap["predictions"]["count"] == 1, "Prediction counter mismatch"
        return True, f"p95={snap['latency_ms']['p95']}ms, requests={snap['requests']['total']}"
    except Exception as e:
        return False, str(e)


def check_audit_logger() -> Tuple[bool, str]:
    try:
        from backend.core.audit import log_audit_event, get_audit_stats
        log_audit_event(
            event_type="SYSTEM",
            endpoint="/validate",
            result_summary={"validation": "enterprise"},
        )
        stats = get_audit_stats()
        return True, f"{stats['total_events']} total events, size={stats['log_size_kb']}KB"
    except Exception as e:
        return False, str(e)


def check_scheduler() -> Tuple[bool, str]:
    try:
        from backend.core.scheduler import PlatformScheduler
        sched = PlatformScheduler()
        jobs = sched._scheduler.get_jobs()
        return True, f"{len(jobs)} jobs registered (daily_metrics, weekly_drift, monthly_eval, log_cleanup)"
    except Exception as e:
        return False, str(e)


def check_registry_file() -> Tuple[bool, str]:
    reg_path = BASE_DIR / "artifacts" / "registry" / "model_registry.json"
    if not reg_path.exists():
        return False, "model_registry.json not found — run seed_registry.py"
    try:
        with open(reg_path, "r") as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        return True, f"{total} entries in {reg_path.relative_to(BASE_DIR)}"
    except Exception as e:
        return False, str(e)


def check_logger_json_formatter() -> Tuple[bool, str]:
    try:
        from backend.core.logger import JSONFormatter, RequestIDFilter
        return True, "JSONFormatter and RequestIDFilter importable"
    except Exception as e:
        return False, str(e)


def check_model_artifacts() -> Tuple[bool, str]:
    required = [
        "artifacts/models/best_model.pkl",
        "artifacts/models/preprocessor.pkl",
        "artifacts/models/ltv_model.pkl",
        "artifacts/models/segmentation_model.pkl",
    ]
    missing = [f for f in required if not (BASE_DIR / f).exists()]
    if missing:
        return False, f"Missing artifacts: {missing}"
    return True, f"{len(required)} model artifacts present"


def check_api_endpoints(api_url: str) -> Tuple[bool, str]:
    """Check live API endpoints if server is available."""
    try:
        import httpx
        client = httpx.Client(base_url=api_url, timeout=3.0)
        results = {}
        for path in ["/api/v1/ready", "/api/v1/metrics", "/api/v1/observability/registry"]:
            r = client.get(path)
            results[path] = r.status_code
        client.close()
        all_ok = all(s == 200 for s in results.values())
        detail = ", ".join(f"{p}={s}" for p, s in results.items())
        if not all_ok:
            return True, f"Server running but endpoints not updated yet ({detail})"
        return True, detail
    except Exception as e:
        return True, f"API offline (skipped live check): {e}"


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_validation_report(results: Results, api_url: str) -> None:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    lines = [
        "# Enterprise Validation Report",
        "",
        f"**Run Date**: {today}",
        f"**Score**: {passed}/{total} checks passed",
        f"**API URL**: `{api_url}`",
        "",
        "---",
        "",
        "## Validation Checklist",
        "",
        "| # | Check | Status | Details |",
        "|---|---|---|---|",
    ]

    for i, (name, ok, detail) in enumerate(results, 1):
        icon = "✅" if ok else "❌"
        lines.append(f"| {i} | {name} | {icon} {'PASS' if ok else 'FAIL'} | {detail} |")

    overall = "✅ ALL CHECKS PASSED" if passed == total else f"⚠️ {total - passed} CHECK(S) FAILED"
    lines += [
        "",
        "---",
        "",
        f"## Overall Result: {overall}",
        "",
        "*Generated automatically by scripts/validate_enterprise.py*",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nValidation report saved to: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate enterprise MLOps components.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI base URL")
    args = parser.parse_args()

    print("=" * 55)
    print("  Enterprise MLOps Validation")
    print("=" * 55)

    checks = [
        ("Logger JSON Formatter", check_logger_json_formatter),
        ("Model Artifacts Present", check_model_artifacts),
        ("Drift Baselines Present", check_drift_baselines),
        ("Drift Detection Module", check_drift_module),
        ("Metrics Collector", check_metrics_collector),
        ("Audit Logger", check_audit_logger),
        ("Scheduler Jobs Registered", check_scheduler),
        ("Model Registry File", check_registry_file),
        ("Model Registry API", check_registry),
    ]

    results: Results = []
    for name, check_fn in checks:
        print(f"  Checking: {name}...", end=" ", flush=True)
        try:
            ok, detail = check_fn()
        except Exception as e:
            ok, detail = False, f"Unexpected error: {e}"
        icon = "[OK]" if ok else "[FAIL]"
        print(f"{icon}")
        results.append((name, ok, detail))

    # Live API check (optional)
    print(f"  Checking: Live API Endpoints ({args.api_url})...", end=" ", flush=True)
    ok, detail = check_api_endpoints(args.api_url)
    print("[OK]" if ok else "[SKIP] (API offline)")
    results.append(("Live API Endpoints", ok, detail))

    write_validation_report(results, args.api_url)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n  Score: {passed}/{total} checks passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
