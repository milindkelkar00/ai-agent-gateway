"""
Sliding window rate limiter.
Swap in Redis for production — the interface stays the same.
"""
import time
from collections import defaultdict, deque
from fastapi import HTTPException
from config.settings import settings


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._windows: dict[str, deque] = defaultdict(deque)

    def check(self, api_key: str) -> None:
        now = time.time()
        window = self._windows[api_key]

        while window and window[0] < now - self.window:
            window.popleft()

        if len(window) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests/minute."
            )

        window.append(now)


rate_limiter = RateLimiter(max_requests=settings.max_requests_per_minute)
