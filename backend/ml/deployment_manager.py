"""
Canary & Blue/Green Deployment Manager.

Manages deployment state transitions (development -> testing -> staging -> production -> archived),
gradual Canary traffic splitting (10% -> 25% -> 50% -> 100%), Blue/Green traffic switching,
and automated rollback triggers.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.logger import logger
from backend.ml.registry import get_production_model, list_versions, promote, rollback

BASE_DIR = Path(__file__).resolve().parents[2]
DEPLOYMENT_STATE_PATH = BASE_DIR / "logs" / "deployment_state.json"
_state_lock = threading.Lock()


class DeploymentManager:
    """Manages Canary splits, Blue/Green environments, and model promotions."""

    def __init__(self) -> None:
        self.state_path = DEPLOYMENT_STATE_PATH
        self._init_state()

    def _init_state(self) -> None:
        if not self.state_path.exists():
            default_state = {
                "active_environment": "blue",
                "blue_model_version": "v1.0.0",
                "green_model_version": "v1.0.0",
                "canary": {
                    "enabled": False,
                    "target_version": None,
                    "traffic_percentage": 0,
                    "stage": "0%",
                    "started_at": None,
                },
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._save_state(default_state)

    def _load_state(self) -> Dict[str, Any]:
        with _state_lock:
            if not self.state_path.exists():
                self._init_state()
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {
                    "active_environment": "blue",
                    "blue_model_version": "v1.0.0",
                    "green_model_version": "v1.0.0",
                    "canary": {"enabled": False, "traffic_percentage": 0},
                }

    def _save_state(self, state: Dict[str, Any]) -> None:
        state["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
        with _state_lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

    def get_deployment_status(self) -> Dict[str, Any]:
        """Return overall deployment status summary."""
        state = self._load_state()
        prod_churn = get_production_model("churn")
        prod_version = prod_churn["version"] if prod_churn else "v1.0.0"

        return {
            "environment": state.get("active_environment", "blue"),
            "production_model_version": prod_version,
            "blue_environment_version": state.get("blue_model_version", "v1.0.0"),
            "green_environment_version": state.get("green_model_version", "v1.0.0"),
            "canary": state.get("canary", {}),
            "last_updated": state.get("last_updated"),
        }

    def update_canary(self, target_version: str, percentage: int) -> Dict[str, Any]:
        """
        Advance Canary deployment percentage (e.g. 10%, 25%, 50%, 100%).

        If percentage == 100, target_version is automatically promoted to production.
        """
        state = self._load_state()

        if percentage <= 0:
            state["canary"] = {
                "enabled": False,
                "target_version": None,
                "traffic_percentage": 0,
                "stage": "0%",
                "started_at": None,
            }
            logger.info("DeploymentManager: Canary deployment cancelled/disabled.")
        elif percentage < 100:
            state["canary"] = {
                "enabled": True,
                "target_version": target_version,
                "traffic_percentage": percentage,
                "stage": f"{percentage}%",
                "started_at": state.get("canary", {}).get("started_at") or datetime.now(tz=timezone.utc).isoformat(),
            }
            logger.info(f"DeploymentManager: Canary advanced to {percentage}% for {target_version}")
        else:
            # 100% -> Complete Canary & promote to Production
            promote("churn", target_version, "production")
            state["canary"] = {
                "enabled": False,
                "target_version": target_version,
                "traffic_percentage": 100,
                "stage": "100% (Completed)",
                "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            if state["active_environment"] == "blue":
                state["blue_model_version"] = target_version
            else:
                state["green_model_version"] = target_version
            logger.info(f"DeploymentManager: Canary completed! {target_version} promoted to production.")

        self._save_state(state)
        return self.get_deployment_status()

    def switch_bluegreen(self) -> Dict[str, Any]:
        """Switch active environment between Blue and Green instantly."""
        state = self._load_state()
        current_env = state.get("active_environment", "blue")
        new_env = "green" if current_env == "blue" else "blue"

        target_version = state.get(f"{new_env}_model_version", "v1.0.0")
        promote("churn", target_version, "production")

        state["active_environment"] = new_env
        self._save_state(state)
        logger.info(f"DeploymentManager: Switched environment to {new_env.upper()} ({target_version})")
        return self.get_deployment_status()

    def execute_rollback(self, model_name: str = "churn") -> Dict[str, Any]:
        """Trigger instant rollback to previous production model."""
        res = rollback(model_name)
        state = self._load_state()
        if res:
            new_version = res["version"]
            env = state.get("active_environment", "blue")
            state[f"{env}_model_version"] = new_version
            state["canary"] = {"enabled": False, "traffic_percentage": 0, "stage": "Rolled Back"}
            self._save_state(state)
            logger.info(f"DeploymentManager: Rolled back {model_name} to {new_version}")

        return {
            "rollback_status": "success" if res else "no_previous_version",
            "active_version": res["version"] if res else "unchanged",
            "deployment_status": self.get_deployment_status(),
        }


# Global DeploymentManager instance
deployment_manager = DeploymentManager()
