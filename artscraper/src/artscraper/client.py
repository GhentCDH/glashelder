"""HTTP client, request throttling and retries.

Ported from `glasgetter/src/client.rs`.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import httpx

USER_AGENT = "glashelder-artscraper/0.1 (sysadmin.ghentcdh@ugent.be)"
MAX_ATTEMPTS = 3
INITIAL_BACKOFF = 1.0
WINDOW = 30
"""Requests kept to work out the current rate."""

logger = logging.getLogger(__name__)

# httpx logs every single request at INFO, which buries the progress view
logging.getLogger("httpx").setLevel(logging.WARNING)


class RestrictedError(Exception):
    """The source refuses to share this artwork."""


@dataclass
class Stats:
    """Live counters for one backend, read by the progress view."""

    items: int = 0
    requests: int = 0
    retries: int = 0
    errors: int = 0
    workers: int = 0
    """Workers this backend was given, whether busy or not."""
    in_flight: int = 0
    """Requests waiting on the network right now."""
    last_latency: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    latency_total: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    """Completion times of the last few requests, for the current rate."""

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def idle(self) -> int:
        return max(0, self.workers - self.in_flight)

    @property
    def rate(self) -> float:
        """Average requests per second over the whole run."""
        return self.requests / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def current_rate(self) -> float:
        """Requests per second over the last few requests."""
        if len(self.recent) < 2:
            return 0.0
        span = self.recent[-1] - self.recent[0]
        return (len(self.recent) - 1) / span if span > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return self.latency_total / self.requests if self.requests else 0.0

    def finished(self, latency: float) -> None:
        """Record one completed request."""
        self.last_latency = latency
        self.latency_total += latency
        self.recent.append(time.monotonic())


class Throttle:
    """Cap both in-flight requests and requests per second.

    One per backend, so a slow source cannot starve a fast one.
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


def make_client(concurrency: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        # httpx sets Accept-Encoding to what it can actually decode; overriding it
        # with brotli or zstd hands us compressed bytes we cannot read.
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=concurrency),
    )


async def fetch_bytes(
    client: httpx.AsyncClient, url: str, throttle: Throttle, stats: Stats
) -> bytes:
    """Fetch a URL, retrying transient failures with exponential backoff."""
    backoff = INITIAL_BACKOFF
    last: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            stats.retries += 1
            await asyncio.sleep(backoff)
            backoff *= 2

        async with throttle:
            stats.requests += 1
            stats.in_flight += 1
            started = time.monotonic()
            try:
                response = await client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                continue
            finally:
                stats.in_flight -= 1
                stats.finished(time.monotonic() - started)

        if response.is_success:
            return response.content

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                await asyncio.sleep(float(retry_after))
            last = httpx.HTTPStatusError(
                f"rate limited by {url}", request=response.request, response=response
            )
            continue

        error = httpx.HTTPStatusError(
            f"{response.status_code} for {url}",
            request=response.request,
            response=response,
        )
        if response.is_server_error:
            last = error
            continue
        # other 4xx are not worth retrying
        stats.errors += 1
        raise error

    stats.errors += 1
    raise last or RuntimeError(f"{url}: out of attempts")


async def fetch_text(
    client: httpx.AsyncClient, url: str, throttle: Throttle, stats: Stats
) -> str:
    raw = await fetch_bytes(client, url, throttle, stats)
    return raw.decode("utf-8", errors="replace")
