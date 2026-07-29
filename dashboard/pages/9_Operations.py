"""
Operations & System Telemetry Dashboard Page - Streamlit Dashboard.

Administrator view showing:
  - System health and API status
  - Automated Ingestion Telemetry & Watch Folder status
  - Model registry table with version details
  - Drift monitoring history and current severity
  - Live metrics (request counts, latency, CPU/memory)
  - Scheduler job status
  - Recent audit log entries & import history
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_executive_header, render_kpi_card
from dashboard.utils.api_client import APIClient

BASE_DIR = Path(__file__).resolve().parents[2]
client = APIClient()


def _safe_api_call(fn, *args, **kwargs):
    """Wrap API call with graceful error handling."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


def render_operations_page() -> None:
    render_executive_header(
        title="⚙️ Enterprise Operations & Automated System Status",
        subtitle="Real-time system health, automated ingestion telemetry, model registry, drift monitoring, and scheduler observability.",
        badge_text="Automated MLOps Pipeline v2.4"
    )

    # -------------------------------------------------------------------
    # 1. System Health Strip
    # -------------------------------------------------------------------
    st.subheader("🏥 System Health & Automated Pipeline Status")
    health_col, ready_col, metrics_col = st.columns(3)

    with health_col:
        health = _safe_api_call(lambda: client._safe_get("/health"))
        status = health.get("status", "unknown") if isinstance(health, dict) else "error"
        color = "#10B981" if status == "healthy" else "#EF4444"
        render_kpi_card(status.upper(), "System Status", border_color=color, trend="Live probe", trend_type="positive" if status == "healthy" else "negative")

    with ready_col:
        ready = _safe_api_call(lambda: client._safe_get("/ready"))
        r_status = ready.get("status", "unknown") if isinstance(ready, dict) else "error"
        render_kpi_card(r_status.upper(), "Liveness Probe", border_color="#3B82F6", trend="Port 8000", trend_type="positive")

    with metrics_col:
        snap = _safe_api_call(lambda: client._safe_get("/metrics"))
        total_req = snap.get("requests", {}).get("total", "N/A") if isinstance(snap, dict) else "N/A"
        render_kpi_card(str(total_req), "Total API Requests", border_color="#F59E0B", trend="Throughput", trend_type="neutral")

    st.divider()

    # -------------------------------------------------------------------
    # 2. Automated Ingestion & Sync Telemetry
    # -------------------------------------------------------------------
    st.subheader("🔄 Automated Ingestion & Sync Telemetry")
    sync_state = _safe_api_call(lambda: client._safe_get("/ingest/state"))
    if not isinstance(sync_state, dict) or "error" in sync_state:
        # Fallback local load
        state_file = BASE_DIR / "data" / "sync_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                sync_state = json.load(f)
        else:
            sync_state = {}

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        last_sync = sync_state.get("last_auto_sync", "Never")
        last_sync_str = last_sync[:19].replace("T", " ") if isinstance(last_sync, str) and "T" in last_sync else str(last_sync)
        render_kpi_card(last_sync_str if last_sync else "Pending", "Last Automatic Sync", border_color="#6366F1", trend="PostgreSQL / Watch", trend_type="positive")

    with s_col2:
        proc_today = sync_state.get("records_processed_today", 0)
        render_kpi_card(f"{proc_today:,}", "Records Processed Today", border_color="#10B981", trend="Auto-Scored", trend_type="positive")

    with s_col3:
        last_pred = sync_state.get("last_prediction_time", "Active")
        last_pred_str = last_pred[:19].replace("T", " ") if isinstance(last_pred, str) and "T" in last_pred else str(last_pred)
        render_kpi_card(last_pred_str if last_pred else "Live", "Last Prediction Time", border_color="#8B5CF6", trend="Pipeline Refresh", trend_type="positive")

    with s_col4:
        last_drift = sync_state.get("last_drift_check", "Daily")
        last_drift_str = last_drift[:10] if isinstance(last_drift, str) and len(last_drift) >= 10 else str(last_drift)
        render_kpi_card(last_drift_str if last_drift else "Daily", "Last Drift / Retrain", border_color="#F59E0B", trend="PSI Monitor", trend_type="neutral")

    st.divider()

    # -------------------------------------------------------------------
    # 3. Model Registry
    # -------------------------------------------------------------------
    st.subheader("📋 Model Registry")
    registry = _safe_api_call(lambda: client._safe_get("/observability/registry"))
    if isinstance(registry, dict) and "error" not in registry:
        rows = []
        for model_name, versions in registry.items():
            for v in versions:
                rows.append({
                    "Model": model_name,
                    "Version": v.get("version"),
                    "Status": v.get("status"),
                    "Type": v.get("model_type"),
                    "Tags": ", ".join(v.get("tags", [])),
                    "Registered": v.get("registered_at", "")[:10],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No models registered yet. Run: python scripts/seed_registry.py")
    else:
        st.warning("Registry unavailable — is the FastAPI backend running?")

    st.divider()

    # -------------------------------------------------------------------
    # 4. Drift Monitoring History
    # -------------------------------------------------------------------
    st.subheader("📊 Drift Monitoring History")
    drift_dir = BASE_DIR / "reports" / "drift"
    if drift_dir.exists():
        history_files = sorted(drift_dir.glob("*.json"), reverse=True)
        if history_files:
            with open(history_files[0], "r", encoding="utf-8") as f:
                latest_drift = json.load(f)
            severity = latest_drift.get("overall_severity", "Unknown")
            severity_colors = {"Normal": "#10B981", "Warning": "#F59E0B", "Critical": "#EF4444"}
            render_kpi_card(severity, f"Latest Drift ({history_files[0].stem})",
                            border_color=severity_colors.get(severity, "#636EFA"))

            trend_rows = []
            for hf in history_files:
                try:
                    with open(hf, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    for feat in d.get("numerical_drift", []):
                        trend_rows.append({"date": hf.stem, "feature": feat["feature"], "psi": feat["psi"]})
                except Exception:
                    pass

            if trend_rows:
                trend_df = pd.DataFrame(trend_rows)
                fig = px.line(
                    trend_df, x="date", y="psi", color="feature",
                    title="PSI Drift History by Feature",
                    template="plotly_dark",
                )
                fig.add_hline(y=0.10, line_dash="dash", line_color="yellow", annotation_text="Warning")
                fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Critical")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift history yet. Run: python scripts/run_drift_check.py")
    else:
        st.info("Drift history directory not found.")

    st.divider()

    # -------------------------------------------------------------------
    # 5. Scheduler Jobs & Ingestion History
    # -------------------------------------------------------------------
    st.subheader("⏰ Scheduled Background Jobs")
    scheduler_data = _safe_api_call(lambda: client._safe_get("/observability/scheduler/jobs"))
    if isinstance(scheduler_data, dict) and "error" not in scheduler_data:
        running = scheduler_data.get("scheduler_running", False)
        st.markdown(f"**Scheduler Status**: {'🟢 Running' if running else '🔴 Stopped'}")
        jobs = scheduler_data.get("jobs", [])
        if jobs:
            st.dataframe(pd.DataFrame(jobs), use_container_width=True)
    else:
        st.warning("Scheduler data unavailable.")

    st.divider()

    st.subheader("📥 Ingestion Import History")
    import_log_path = BASE_DIR / "logs" / "imports.jsonl"
    if import_log_path.exists():
        imp_events = []
        with open(import_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        imp_events.append(json.loads(line.strip()))
                    except Exception:
                        pass
        if imp_events:
            imp_df = pd.DataFrame(list(reversed(imp_events[-30:])))
            st.dataframe(imp_df, use_container_width=True)
        else:
            st.info("No ingestion events logged yet.")
    else:
        st.info("Import log not created yet. Put a CSV in data/incoming/ to test watch folder!")


# Monkey-patch APIClient with a safe_get helper
def _patch_client(client: APIClient) -> APIClient:
    import httpx

    def _safe_get(path: str):
        try:
            url = f"{client.base_url}{path}"
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    client._safe_get = _safe_get
    return client


client = _patch_client(client)

if __name__ == "__main__":
    render_operations_page()
