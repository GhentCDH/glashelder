"""Run backends side by side, stream results to disk, report progress."""

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import zstandard as zstd

from .backends import Context, registry
from .client import Stats, Throttle, make_client
from .model import Artwork

logger = logging.getLogger(__name__)

type Reporter = Callable[[str, Artwork | None, Stats], None]
"""Called with the backend name and every artwork it produces, for progress views."""


def scraped_urls(path: Path) -> dict[str, set[str]]:
    """URLs an earlier, possibly interrupted, run already scraped, per backend."""
    done: dict[str, set[str]] = {}
    if not path.exists():
        return done
    with path.open() as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                done.setdefault(row["source"], set()).add(row["url"])
    return done


def combine_files(*sources: Path, path: Path | None) -> None:
    """Turn multiple JSON, JSONL, or Zst files into a combined output (auto-compresses if path ends in .zst)."""

    # Choose the correct writer based on output extension
    if path:
        is_zst_output = ".zst" in [s.lower() for s in path.suffixes]
        out = (
            zstd.open(path, "wt", encoding="utf-8")
            if is_zst_output
            else path.open("w", encoding="utf-8")
        )
    else:
        out = sys.stdout

    count = 0
    joined = []

    try:
        for file_path in sources:
            suffixes = [s.lower() for s in file_path.suffixes]
            is_zst_input = ".zst" in suffixes
            is_json_input = ".json" in suffixes

            # Read input (compressed or normal)
            file_context = (
                zstd.open(file_path, "rt", encoding="utf-8")
                if is_zst_input
                else file_path.open("r", encoding="utf-8")
            )

            with file_context as f:
                if is_json_input:
                    data = json.load(f)
                    for item in data:
                        artwork = Artwork(**item)
                        joined.append(asdict(artwork))
                        count += 1
                else:
                    for line in f:
                        if line.strip():
                            artwork = Artwork(**json.loads(line))
                            joined.append(asdict(artwork))
                            count += 1

        json.dump(joined, out, ensure_ascii=False, indent=2)

    finally:
        if path:
            out.close()
            logger.info("wrote %d artworks to %s", count, path)


async def scrape(
    names: list[str],
    *,
    jsonl: Path,
    concurrency: int = 10,
    rate: float = 5.0,
    limit: int = 0,
    with_images: bool = True,
    report: Reporter | None = None,
) -> None:
    """Scrape every named backend concurrently, appending to `jsonl`.

    URLs already in that file are skipped, so an interrupted run can simply be
    started again. Nothing is kept in memory beyond the URLs seen so far: a full
    run is millions of records, so `jsonl` is the result.
    """
    backends = registry()
    if unknown := [name for name in names if name not in backends]:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")

    done = scraped_urls(jsonl)
    if done:
        logger.info(
            "resuming: %d artworks already in %s", sum(map(len, done.values())), jsonl
        )

    with jsonl.open("a") as sink:

        async def run(name: str) -> None:
            seen = done.get(name, set())
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
                ctx.stats.items += 1
                sink.write(json.dumps(asdict(artwork), ensure_ascii=False) + "\n")
                if report:
                    report(name, artwork, ctx.stats)
            sink.flush()

        async with make_client(concurrency) as client:
            await asyncio.gather(*(run(name) for name in names))
