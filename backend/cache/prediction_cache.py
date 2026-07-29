"""
Redis & In-Memory Fallback Prediction Cache.

Provides high-throughput prediction caching with TTL support and LRU eviction.
Tracks hits, misses, and hit ratio metrics. Fallback to thread-safe in-memory
dict if Redis is unavailable.
"""

import json
import hashlib
import os
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from backend.core.logger import logger
from backend.core.settings import settings

# Attempt redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class InMemoryLRUCache:
    """Thread-safe in-memory LRU cache with TTL support."""

    def __init__(self, maxsize: int = 1000, default_ttl: int = 3600):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._cache:
                return None
            val, expire_at = self._cache[key]
            if time.time() > expire_at:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return val

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        with self._lock:
            expire_at = time.time() + (ttl if ttl is not None else self.default_ttl)
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expire_at)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class PredictionCache:
    """
    Unified prediction caching service.
    Uses Redis if available, else in-memory LRU fallback.
    """

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.default_ttl = 3600
        self.redis_client: Optional[Any] = None
        self._memory_cache = InMemoryLRUCache(maxsize=2000, default_ttl=self.default_ttl)
        self._lock = threading.Lock()
        self._init_redis()

    def _init_redis(self) -> None:
        if not REDIS_AVAILABLE:
            logger.info("PredictionCache: redis module not installed. Using in-memory LRU fallback.")
            return

        redis_host = getattr(settings, "REDIS_HOST", os.getenv("REDIS_HOST", "localhost"))
        redis_port = getattr(settings, "REDIS_PORT", int(os.getenv("REDIS_PORT", 6379)))

        try:
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            client.ping()
            self.redis_client = client
            logger.info(f"PredictionCache: Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"PredictionCache: Redis unavailable ({e}). Using in-memory LRU fallback.")
            self.redis_client = None

    def _hash_payload(self, payload: Dict[str, Any], model_version: str = "v1.0.0") -> str:
        """Create a deterministic SHA256 key from prediction features + model version."""
        raw = json.dumps({"payload": payload, "version": model_version}, sort_keys=True)
        return "pred_cache:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_prediction(self, payload: Dict[str, Any], model_version: str = "v1.0.0") -> Optional[Dict[str, Any]]:
        """Retrieve prediction result from cache."""
        key = self._hash_payload(payload, model_version)
        cached_str: Optional[str] = None

        if self.redis_client:
            try:
                cached_str = self.redis_client.get(key)
            except Exception:
                cached_str = self._memory_cache.get(key)
        else:
            cached_str = self._memory_cache.get(key)

        with self._lock:
            if cached_str is not None:
                self.hits += 1
                return json.loads(cached_str)
            else:
                self.misses += 1
                return None

    def set_prediction(
        self,
        payload: Dict[str, Any],
        result: Dict[str, Any],
        model_version: str = "v1.0.0",
        ttl: Optional[int] = None,
    ) -> None:
        """Store prediction result in cache."""
        key = self._hash_payload(payload, model_version)
        val_str = json.dumps(result, ensure_ascii=False)
        ttl_seconds = ttl if ttl is not None else self.default_ttl

        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl_seconds, val_str)
            except Exception:
                self._memory_cache.set(key, val_str, ttl_seconds)
        else:
            self._memory_cache.set(key, val_str, ttl_seconds)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance metrics."""
        with self._lock:
            total = self.hits + self.misses
            ratio = (self.hits / total) if total > 0 else 0.0
            backend = "redis" if self.redis_client is not None else "in_memory_lru"
            cached_items = self._memory_cache.size() if self.redis_client is None else "managed_by_redis"

            return {
                "backend": backend,
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_ratio": round(ratio, 4),
                "hit_percentage": f"{ratio * 100:.2f}%",
                "in_memory_items": cached_items,
            }

    def clear(self) -> None:
        """Clear cache entries."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self._memory_cache.clear()
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception:
                    pass


# Global singleton
prediction_cache = PredictionCache()
