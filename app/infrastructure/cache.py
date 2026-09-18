"""Redis 缓存适配器及进程内降级实现。

读数据流：业务服务 → Redis →（未命中/异常）内存缓存 → 返回 ``None`` 让业务回源。
写数据流：业务服务 → Redis ``SETEX``；Redis 异常时 → 内存 TTL 字典。
注意：缓存不是事实源，MySQL 才保存不可丢失的数据。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Cache:
    """Redis 优先、进程内缓存兜底的统一接口。

    Redis 宕机会降低性能并失去跨实例会话共享，但不会让核心 API 完全不可用。
    """

    def __init__(self, settings: Settings):
        """尝试连接 Redis；连接失败时记录告警并保持 memory 后端。"""

        self.default_ttl = settings.redis_ttl_seconds
        self.session_ttl = settings.session_ttl_seconds
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.RLock()
        self._redis: Any = None
        self.backend_name = "memory"

        if settings.redis_url:
            # 数据流：REDIS_URL → 连接与 PING → 成功切换 backend_name=redis。
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
        """统一 Key 前缀，避免与同一 Redis 中的其他应用冲突。"""

        return f"agent-platform:{namespace}:{key}"

    def get_json(self, key: str) -> Any | None:
        """读取 JSON 值；Redis 未命中后检查本进程缓存。"""

        # 数据流：key → Redis GET → memory fallback → JSON 反序列化 → Service。
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
                    # 惰性淘汰：读取时发现过期才删除，避免额外清理线程。
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
        """序列化并写入带 TTL 的缓存；Redis 失败才写内存。"""

        # 数据流：Python 对象 → JSON → Redis SETEX / memory(expire_at, raw)。
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
        """按命名空间模式失效缓存，返回删除数量。"""

        # 使用 SCAN 而不是 KEYS，避免生产 Redis 被一次全量扫描阻塞。
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
        """从热缓存读取最近会话。"""

        # 数据流：session_id → namespaced key → list[role/content]。
        key = self.namespaced("session", session_id)
        value = self.get_json(key)
        return value if isinstance(value, list) else None

    def set_history(self, session_id: str, history: list[dict[str, str]]) -> None:
        """只缓存最近 20 条消息，并使用更长的会话 TTL。"""

        key = self.namespaced("session", session_id)
        self.set_json(key, history[-20:], ttl=self.session_ttl)

    def health(self) -> bool:
        """内存后端天然可用；Redis 后端通过 PING 检测。"""

        if self._redis is None:
            return True
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def close(self) -> None:
        """应用退出时释放 Redis 连接池。"""

        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass

    @staticmethod
    def _matches(pattern: str, key: str) -> bool:
        """为内存降级实现最小的前缀通配匹配。"""

        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern
