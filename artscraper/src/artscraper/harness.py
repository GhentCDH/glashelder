"""Run backends side by side, stream results to disk, report progress."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .backends import Context, registry
from .client import Stats, Throttle, make_client
from .model import Artwork

logger = logging.getLogger(__name__)

type Reporter = Callable[[str, Artwork | None, Stats], None]
"""Called with the backend name and every artwork it produces, for progress views."""


def read_jsonl(path: Path) -> list[Artwork]:
    """Load artworks from an earlier, possibly interrupted, run."""
    if not path.exists():
        return []
    with path.open() as file:
        return [Artwork(**json.loads(line)) for line in file if line.strip()]


def write_json(artworks: list[Artwork], path: Path | None) -> None:
    payload = json.dumps([asdict(a) for a in artworks], ensure_ascii=False, indent=1)
    if path is None:
        print(payload)
    else:
        path.write_text(payload)
        logger.info("wrote %d artworks to %s", len(artworks), path)


async def scrape(
    names: list[str],
    *,
    jsonl: Path,
    concurrency: int = 10,
    rate: float = 5.0,
    limit: int = 0,
    with_images: bool = True,
    report: Reporter | None = None,
) -> list[Artwork]:
    """Scrape every named backend concurrently.

    Results are appended to `jsonl` as they arrive, and URLs already in that file
    are skipped, so an interrupted run can simply be started again.
    """
    backends = registry()
    if unknown := [name for name in names if name not in backends]:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")

    done = read_jsonl(jsonl)
    if done:
        logger.info("resuming: %d artworks already in %s", len(done), jsonl)
    results = {name: [a for a in done if a.source == name] for name in names}

    with jsonl.open("a") as sink:

        async def run(name: str) -> None:
            seen = {a.url for a in results[name]}
            if limit and len(seen) >= limit:
                logger.info(
                    "%s: already has %d artworks, nothing to do", name, len(seen)
                )
                return

            ctx = Context(
                client=client,
                throttle=Throttle(concurrency, rate),
                stats=Stats(workers=concurrency),
                limit=limit - len(seen) if limit else 0,
                with_images=with_images,
                workers=concurrency,
                seen=seen,
            )
            if report:
                report(
                    name, None, ctx.stats
                )  # register the backend before its first item
            async for artwork in backends[name].stream(ctx):
                results[name].append(artwork)
                ctx.stats.items += 1
                sink.write(json.dumps(asdict(artwork), ensure_ascii=False) + "\n")
                if report:
                    report(name, artwork, ctx.stats)
            sink.flush()

        async with make_client(concurrency) as client:
            await asyncio.gather(*(run(name) for name in names))

    return [artwork for name in names for artwork in results[name]]
