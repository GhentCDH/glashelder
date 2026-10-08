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
from ..model import Artwork, merge

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

    async def complete(
        self, ctx: Context, artworks: list[Artwork]
    ) -> AsyncIterator[Artwork]:
        """Yield artworks with their missing fields filled in from the source.

        Takes a batch, so a source with a bulk API can answer in a few requests.
        Artworks the source cannot complete are left out. Without an override,
        nothing is completed.
        """
        for artwork in ():
            yield artwork


def completes(backend: Backend) -> bool:
    """Whether a backend overrides `Backend.complete`."""
    return type(backend).complete is not Backend.complete


async def sitemap_locs(ctx: Context, url: str) -> list[str]:
    """Return every `<loc>` in a sitemap or sitemap index."""
    raw = await fetch_bytes(ctx.client, url, ctx.throttle, ctx.stats)
    root = ElementTree.fromstring(raw.lstrip(b"\xef\xbb\xbf"))
    return [
        loc.text.strip()
        for loc in root.iter(f"{SITEMAP_NS}loc")
        if loc.text is not None
    ]


async def enrich_pool[T](
    ctx: Context,
    items: list[T],
    fetch_one: Callable[[Context, T], Awaitable[Artwork | None]],
    limit: int | None = 0,
) -> AsyncIterator[Artwork]:
    """Run `fetch_one` over `items` with a bounded pool, yielding as results land.

    Per-artwork failures are logged and dropped: one bad page must not kill the run.
    """
    todo = iter(items)
    queue: asyncio.Queue[Artwork | None] = asyncio.Queue(maxsize=ctx.workers)
    done = asyncio.Event()

    async def worker() -> None:
        for item in todo:  # generators are safe to share between workers
            if done.is_set():
                return
            try:
                artwork = await fetch_one(ctx, item)
            except RestrictedError:
                logger.debug("%s is restricted", item)
                continue
            except Exception as e:
                logger.warning("skipping %s: %s", item, e)
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


def physical(*parts: object) -> str | None:
    """A `physical_description` from the parts a source has, e.g. type, medium, size.

    Empty and repeated parts (some sources give the same text as type and medium)
    are left out.
    """
    texts = [
        text
        for part in parts
        if isinstance(part, str) and (text := " ".join(part.split()))
    ]
    folded = [text.casefold() for text in texts]
    unique = [text for i, text in enumerate(texts) if folded[i] not in folded[:i]]
    return "; ".join(unique) or None


def complete_pages(
    ctx: Context,
    artworks: list[Artwork],
    fetch_one: Callable[[Context, str], Awaitable[Artwork | None]],
) -> AsyncIterator[Artwork]:
    """`Backend.complete` for sources with one page per artwork at `artwork.url`.

    `fetch_one` is the scrape's own page fetcher, so a scrape and a completion
    read the same fields.
    """

    async def complete_one(ctx: Context, artwork: Artwork) -> Artwork | None:
        found = await fetch_one(ctx, artwork.url)
        return merge(artwork, found) if found else None

    return enrich_pool(ctx, artworks, complete_one)


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
    from .cle import Cleveland
    from .kc import Kunstcollectie
    from .kikirpa import Kikirpa
    from .louvre import Louvre
    from .met import MetMuseum
    from .nga import Nga
    from .uffizi import Uffizi
    from .wikiart import WikiArt
    from .wm import Wereldmuseum

    return {
        backend.name: backend
        for backend in (
            WikiArt(),
            Kikirpa(),
            Wereldmuseum(),
            MetMuseum(),
            Nga(),
            Louvre(),
            Uffizi(),
            Cleveland(),
            Kunstcollectie(),
        )
    }
