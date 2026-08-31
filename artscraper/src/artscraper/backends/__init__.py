"""Backends, one per data source, and the glue the harness talks to."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol
from xml.etree import ElementTree

import httpx

from ..client import RestrictedError, Stats, Throttle, fetch_bytes
from ..model import Artwork

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Everything a backend needs to do its work."""

    client: httpx.AsyncClient
    throttle: Throttle
    stats: Stats
    limit: int = 0
    """Maximum artworks to yield, 0 for no limit."""
    with_images: bool = True
    """Fetch each artwork's page to find its image URL."""
    workers: int = 10
    """In-flight artwork page fetches; the throttle still caps the request rate."""
    seen: set[str] = field(default_factory=set)
    """URLs already scraped in an earlier run, skipped."""


class Backend(Protocol):
    name: str

    def stream(self, ctx: Context) -> AsyncIterator[Artwork]: ...


async def sitemap_locs(ctx: Context, url: str) -> list[str]:
    """Return every `<loc>` in a sitemap or sitemap index."""
    raw = await fetch_bytes(ctx.client, url, ctx.throttle, ctx.stats)
    root = ElementTree.fromstring(raw.lstrip(b"\xef\xbb\xbf"))
    return [
        loc.text.strip()
        for loc in root.iter(f"{SITEMAP_NS}loc")
        if loc.text is not None
    ]


async def enrich_pool(
    ctx: Context,
    urls: list[str],
    fetch_one: Callable[[Context, str], Awaitable[Artwork | None]],
) -> AsyncIterator[Artwork]:
    """Run `fetch_one` over `urls` with a bounded pool, yielding as results land.

    Per-artwork failures are logged and dropped: one bad page must not kill the run.
    """
    todo = (url for url in urls if url not in ctx.seen)
    queue: asyncio.Queue[Artwork | None] = asyncio.Queue(maxsize=ctx.workers)
    done = asyncio.Event()

    async def worker() -> None:
        for url in todo:  # generators are safe to share between workers
            if done.is_set():
                return
            try:
                artwork = await fetch_one(ctx, url)
            except RestrictedError:
                logger.debug("%s is restricted", url)
                continue
            except Exception as e:
                logger.warning("skipping %s: %s", url, e)
                continue
            if artwork is not None:
                await queue.put(artwork)

    async def run() -> None:
        await asyncio.gather(*(worker() for _ in range(ctx.workers)))
        await queue.put(None)  # sentinel: no more results coming

    runner = asyncio.create_task(run())
    yielded = 0
    try:
        while (artwork := await queue.get()) is not None:
            yield artwork
            yielded += 1
            if ctx.limit and yielded >= ctx.limit:
                return
    finally:
        done.set()
        runner.cancel()


def registry() -> dict[str, Backend]:
    """All configured backends by name."""
    from .kikirpa import Kikirpa
    from .wikiart import WikiArt

    return {backend.name: backend for backend in (WikiArt(), Kikirpa())}
