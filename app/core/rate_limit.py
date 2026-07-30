"""Simple in-memory sliding-window rate limiter."""
from __future__ import annotations

import time
from collections import defaultdict

from app.core.config import settings


class TokenBucket:
    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max_rpm
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        if self._max_rpm <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        bucket = self._buckets[key]
        bucket[:] = [t for t in bucket if t > window_start]
        if len(bucket) >= self._max_rpm:
            return False
        bucket.append(now)
        return True

    @property
    def remaining(self, key: str) -> int:
        if self._max_rpm <= 0:
            return 0
        now = time.monotonic()
        window_start = now - 60.0
        bucket = self._buckets[key]
        bucket[:] = [t for t in bucket if t > window_start]
        return max(0, self._max_rpm - len(bucket))


rate_limiter = TokenBucket(settings.rate_limit_per_minute)
