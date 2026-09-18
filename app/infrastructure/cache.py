from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Cache:
    """Redis-first cache with a transparent in-process fallback.

    Redis is an optimization, not the system of record. A Redis outage therefore
    degrades performance and cross-instance session sharing, but does not break the API.
    """

    def __init__(self, settings: Settings):
        self.default_ttl = settings.redis_ttl_seconds
        self.session_ttl = settings.session_ttl_seconds
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.RLock()
        self._redis: Any = None
        self.backend_name = "memory"

        if settings.redis_url:
            try:
                import redis

                client = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    health_check_interval=30,
                )
                client.ping()
                self._redis = client
                self.backend_name = "redis"
            except Exception as exc:
                logger.warning("Redis unavailable; using memory cache: %s", exc)

    @staticmethod
    def namespaced(namespace: str, key: str) -> str:
        return f"agent-platform:{namespace}:{key}"

    def get_json(self, key: str) -> Any | None:
        raw: str | None = None
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
            except Exception as exc:
                logger.warning("Redis read failed; using memory fallback: %s", exc)

        if raw is None:
            with self._lock:
                item = self._memory.get(key)
                if item:
                    expires_at, raw = item
                    if expires_at <= time.time():
                        self._memory.pop(key, None)
                        raw = None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl
        raw = json.dumps(value, ensure_ascii=False, default=str)
        stored_in_redis = False
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, raw)
                stored_in_redis = True
            except Exception as exc:
                logger.warning("Redis write failed; using memory fallback: %s", exc)
        if not stored_in_redis:
            with self._lock:
                self._memory[key] = (time.time() + ttl, raw)

    def delete_pattern(self, pattern: str) -> int:
        deleted = 0
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(match=pattern, count=200))
                if keys:
                    deleted += int(self._redis.delete(*keys))
            except Exception as exc:
                logger.warning("Redis invalidation failed: %s", exc)
        with self._lock:
            for key in list(self._memory):
                if self._matches(pattern, key):
                    self._memory.pop(key, None)
                    deleted += 1
        return deleted

    def get_history(self, session_id: str) -> list[dict[str, str]] | None:
        key = self.namespaced("session", session_id)
        value = self.get_json(key)
        return value if isinstance(value, list) else None

    def set_history(self, session_id: str, history: list[dict[str, str]]) -> None:
        key = self.namespaced("session", session_id)
        self.set_json(key, history[-20:], ttl=self.session_ttl)

    def health(self) -> bool:
        if self._redis is None:
            return True
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass

    @staticmethod
    def _matches(pattern: str, key: str) -> bool:
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern

