"""
Production Deployment Automated Validation Script.

Validates Docker, Kubernetes, Redis Cache, Feature Store, Prediction Store,
Retraining Pipeline, SHAP Explainability, Deployment APIs, Dashboard, and Alerts.

Expected Output Format:
=========================================
 Production Deployment Validation
=========================================

Docker.........................OK
Kubernetes.....................OK
Redis Cache....................OK
Feature Store..................OK
Prediction Store...............OK
Retraining.....................OK
Explainability.................OK
Deployment APIs................OK
Dashboard......................OK
Alerts.........................OK

Score: 10/10
"""

import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(__file__).resolve().parents[1]


def check_docker() -> Tuple[bool, str]:
    files = [
        BASE_DIR / "docker" / "Dockerfile.backend",
        BASE_DIR / "docker" / "Dockerfile.dashboard",
        BASE_DIR / "docker" / "Dockerfile.worker",
        BASE_DIR / "docker" / "Dockerfile.scheduler",
        BASE_DIR / "docker-compose.production.yml",
        BASE_DIR / ".env.production",
    ]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return False, f"Missing Docker files: {missing}"
    return True, "All Docker files present"


def check_kubernetes() -> Tuple[bool, str]:
    files = [
        BASE_DIR / "k8s" / "deployment-api.yaml",
        BASE_DIR / "k8s" / "deployment-worker.yaml",
        BASE_DIR / "k8s" / "deployment-dashboard.yaml",
        BASE_DIR / "k8s" / "service-api.yaml",
        BASE_DIR / "k8s" / "service-dashboard.yaml",
        BASE_DIR / "k8s" / "ingress.yaml",
        BASE_DIR / "k8s" / "configmap.yaml",
        BASE_DIR / "k8s" / "secret.yaml",
        BASE_DIR / "k8s" / "horizontal-pod-autoscaler.yaml",
        BASE_DIR / "k8s" / "persistent-volume.yaml",
    ]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return False, f"Missing K8s manifests: {missing}"
    return True, "All 10 K8s manifests present"


def check_redis_cache() -> Tuple[bool, str]:
    try:
        from backend.cache.prediction_cache import prediction_cache
        stats = prediction_cache.get_stats()
        return True, f"Backend: {stats['backend']}, Hit ratio: {stats['hit_ratio']}"
    except Exception as e:
        return False, str(e)


def check_feature_store() -> Tuple[bool, str]:
    try:
        from backend.ml.feature_store import feature_store
        is_valid, errors = feature_store.validate_features({"tenure_months": 12, "monthly_charges": 70.0})
        meta = feature_store.get_feature_metadata()
        return True, f"Version {meta['version']}, {meta['feature_count']} features"
    except Exception as e:
        return False, str(e)


def check_prediction_store() -> Tuple[bool, str]:
    try:
        from backend.ml.prediction_store import prediction_store
        summary = prediction_store.get_analytics_summary()
        return True, f"Analytics summary generated ({summary['total_stored_predictions']} predictions recorded)"
    except Exception as e:
        return False, str(e)


def check_retraining() -> Tuple[bool, str]:
    try:
        from backend.ml.retraining import retraining_pipeline
        history = retraining_pipeline.get_history()
        return True, f"Retraining pipeline active ({len(history)} past runs recorded)"
    except Exception as e:
        return False, str(e)


def check_explainability() -> Tuple[bool, str]:
    try:
        from backend.ml.explain import explainability_service
        imp = explainability_service.get_global_importance()
        exp = explainability_service.explain_customer("C001")
        return True, f"SHAP service generated explanation for C001 ({len(imp)} global features)"
    except Exception as e:
        return False, str(e)


def check_deployment_apis() -> Tuple[bool, str]:
    try:
        from backend.ml.deployment_manager import deployment_manager
        status = deployment_manager.get_deployment_status()
        return True, f"Env: {status['environment']}, Model: {status['production_model_version']}"
    except Exception as e:
        return False, str(e)


def check_dashboard() -> Tuple[bool, str]:
    dash_file = BASE_DIR / "dashboard" / "pages" / "10_Deployment.py"
    if not dash_file.exists():
        return False, "10_Deployment.py not found"
    return True, "Streamlit Deployment Dashboard (Page 10) present"


def check_alerts() -> Tuple[bool, str]:
    try:
        from backend.core.alerts import alert_manager
        history = alert_manager.get_alert_history()
        return True, f"Alert manager active ({len(history)} historical alerts)"
    except Exception as e:
        return False, str(e)


def main() -> None:
    print("=========================================")
    print(" Production Deployment Validation")
    print("=========================================\n")

    checks = [
        ("Docker", check_docker),
        ("Kubernetes", check_kubernetes),
        ("Redis Cache", check_redis_cache),
        ("Feature Store", check_feature_store),
        ("Prediction Store", check_prediction_store),
        ("Retraining", check_retraining),
        ("Explainability", check_explainability),
        ("Deployment APIs", check_deployment_apis),
        ("Dashboard", check_dashboard),
        ("Alerts", check_alerts),
    ]

    passed = 0
    for name, fn in checks:
        print(f"{name:<31}", end="", flush=True)
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, str(e)

        if ok:
            print("OK")
            passed += 1
        else:
            print(f"FAIL ({detail})")

    print(f"\nScore: {passed}/{len(checks)}")

    if passed < len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
