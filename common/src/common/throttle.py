"""Politeness for anything that talks to someone else's server."""

import asyncio
import time


class Throttle:
    """Cap both in-flight requests and requests per second.

    The concurrency cap backs off by itself when a server slows down, the rate cap
    keeps a fast server from being hit harder than intended. Use one per server, so
    a slow one cannot starve a fast one.
    """

    def __init__(self, concurrency: int, rate: float):
        self._sem = asyncio.Semaphore(concurrency)
        self._interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def __aenter__(self) -> None:
        await self._sem.acquire()
        if self._interval:
            async with self._lock:
                now = time.monotonic()
                wait = self._next - now
                self._next = max(now, self._next) + self._interval
            if wait > 0:
                await asyncio.sleep(wait)

    async def __aexit__(self, *_exc) -> None:
        self._sem.release()

    def refund(self) -> None:
        """Give back the last request's place in the rate, for one that cost the
        server next to nothing (like a 404)."""
        self._next = max(time.monotonic(), self._next - self._interval)
