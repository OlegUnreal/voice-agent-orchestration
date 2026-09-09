"""In-process TTL cache for identical turns (gateway response cache)."""

from __future__ import annotations

import time
from threading import Lock


class TtlCache:
    def __init__(self, ttl_s: float = 600.0):
        self.ttl_s = ttl_s
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            row = self._data.get(key)
            if not row:
                self.misses += 1
                return None
            ts, value = row
            if time.time() - ts > self.ttl_s:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}
