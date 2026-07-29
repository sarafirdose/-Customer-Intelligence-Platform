"""
REST API Client for Customer Intelligence Dashboard.

Connects to the FastAPI backend service (defaulting to http://localhost:8000).
"""

import os
from typing import Any, Dict, List
import httpx

from backend.core.logger import logger

# Base URL for API requests
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")


class APIClient:
    """
    HTTP Client wrapper for Customer REST APIs.
    """

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def get_customer_intelligence(self, customer_id: str) -> Dict[str, Any]:
        """
        Fetch unified intelligence metrics for a single customer.
        """
        try:
            url = f"{self.base_url}/customer/{customer_id}"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": f"Customer ID {customer_id} not found."}
            else:
                return {"error": f"API returned error code {response.status_code}."}
        except Exception as e:
            logger.error(f"API Error fetching customer: {e}")
            return {"error": f"Connection to API failed: {str(e)}"}

    def get_customer_ltv(self, customer_id: str) -> Dict[str, Any]:
        """
        Fetch LTV predictions and expected lifespans.
        """
        try:
            url = f"{self.base_url}/customer/{customer_id}/ltv"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "LTV lookup failed."}
        except Exception as e:
            return {"error": str(e)}

    def get_customer_segment(self, customer_id: str) -> Dict[str, Any]:
        """
        Fetch K-Means cluster details.
        """
        try:
            url = f"{self.base_url}/customer/{customer_id}/segment"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "Segmentation lookup failed."}
        except Exception as e:
            return {"error": str(e)}

    def get_customer_score(self, customer_id: str) -> Dict[str, Any]:
        """
        Fetch unified Customer Intelligence Score (0-100).
        """
        try:
            url = f"{self.base_url}/customer/{customer_id}/intelligence"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "Intelligence score lookup failed."}
        except Exception as e:
            return {"error": str(e)}

    def get_customer_recommendations(self, customer_id: str) -> Dict[str, Any]:
        """
        Fetch dynamic retention actions.
        """
        try:
            url = f"{self.base_url}/customer/{customer_id}/recommendations"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"error": "Recommendations lookup failed."}
        except Exception as e:
            return {"error": str(e)}

    def batch_score_customers(self, customer_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Trigger batch customer scoring.
        """
        try:
            url = f"{self.base_url}/customers/batch_intelligence"
            payload = {"customer_ids": customer_ids}
            response = httpx.post(url, json=payload, timeout=120.0)
            if response.status_code == 200:
                return response.json()
            return [{"error": f"Batch scoring returned code {response.status_code}."}]
        except Exception as e:
            return [{"error": str(e)}]
