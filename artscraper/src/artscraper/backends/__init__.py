"""Backends, one per data source, and the glue the harness talks to."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol
from xml.etree import ElementTree

import httpx

from common.throttle import Throttle

from ..client import RestrictedError, Stats, fetch_bytes
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
    limit: int = 0,
) -> AsyncIterator[Artwork]:
    """Run `fetch_one` over `urls` with a bounded pool, yielding as results land.

    Per-artwork failures are logged and dropped: one bad page must not kill the run.
    """
    todo = iter(urls)
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
            if limit and yielded >= limit:
                return
    finally:
        done.set()
        runner.cancel()


async def stream_sitemaps(
    ctx: Context,
    sitemaps: list[str],
    fetch_one: Callable[[Context, str], Awaitable[Artwork | None]],
    from_url: Callable[[str], Artwork | None],
) -> AsyncIterator[Artwork]:
    """Yield artworks from a list of sitemaps, one sitemap at a time.

    Deliberately not gathering every URL up front: a full run is hundreds of
    thousands of them per source, and only one sitemap's worth needs to be in
    memory at a time.
    """
    yielded = 0
    for sitemap in sitemaps:
        urls = [url for url in await sitemap_locs(ctx, sitemap) if url not in ctx.seen]
        logger.info("%d urls to scrape from %s", len(urls), sitemap)
        remaining = ctx.limit - yielded if ctx.limit else 0

        if ctx.with_images:
            async for artwork in enrich_pool(ctx, urls, fetch_one, remaining):
                yield artwork
                yielded += 1
        else:
            for url in urls:
                if (artwork := from_url(url)) is not None:
                    yield artwork
                    yielded += 1
                if ctx.limit and yielded >= ctx.limit:
                    break

        if ctx.limit and yielded >= ctx.limit:
            return


def registry() -> dict[str, Backend]:
    """All configured backends by name."""
    from .kikirpa import Kikirpa
    from .wikiart import WikiArt

    return {backend.name: backend for backend in (WikiArt(), Kikirpa())}
