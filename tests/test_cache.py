"""
Unit tests for PredictionCache & InMemoryLRUCache.
"""

import time
import pytest
from backend.cache.prediction_cache import InMemoryLRUCache, PredictionCache


def test_in_memory_lru_cache():
    cache = InMemoryLRUCache(maxsize=2, default_ttl=10)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    assert cache.get("k1") == "v1"
    assert cache.get("k2") == "v2"

    # Eviction test
    cache.set("k3", "v3")
    assert cache.get("k1") is None  # k1 evicted
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"


def test_prediction_cache_get_set():
    pc = PredictionCache()
    pc.clear()

    payload = {"tenure_months": 12, "monthly_charges": 70.0}
    res = {"churn_probability": 0.25, "risk_level": "Low"}

    # Miss
    assert pc.get_prediction(payload, "v1.0.0") is None

    # Set & Hit
    pc.set_prediction(payload, res, "v1.0.0")
    cached = pc.get_prediction(payload, "v1.0.0")
    assert cached is not None
    assert cached["churn_probability"] == 0.25

    # Stats
    stats = pc.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert stats["total_requests"] >= 2
