"""Run backends side by side, stream results to disk, report progress."""

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import IO

from common.throttle import Throttle

from .backends import Context, completes, registry
from .client import Stats, make_client
from .index import artwork_of, index, open_text, read
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
    """Join JSON, JSONL or `.zst` scrapes into one JSON list, compressed if `path` ends in `.zst`.

    Without a `path` the list goes to stdout.
    """
    joined = [asdict(artwork) for source in sources for artwork in read(source)]
    with open_text(path, "w") if path else nullcontext(sys.stdout) as out:
        json.dump(joined, out, ensure_ascii=False, indent=2)
    if path:
        logger.info("wrote %d artworks to %s", len(joined), path)


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
            await drain(name, backends[name].stream(ctx), ctx, sink, report)

        async with make_client(concurrency) as client:
            await asyncio.gather(*(run(name) for name in names))


async def drain(
    name: str,
    artworks: AsyncIterator[Artwork],
    ctx: Context,
    sink: IO[str],
    report: Reporter | None,
) -> None:
    """Append every artwork a backend yields to `sink` as one JSON line."""
    if report:
        report(name, None, ctx.stats)  # register the backend before its first item
    async for artwork in artworks:
        ctx.stats.items += 1
        sink.write(json.dumps(asdict(artwork), ensure_ascii=False) + "\n")
        if report:
            report(name, artwork, ctx.stats)
    sink.flush()


def matched_artworks(
    rows: list[dict], artworks: Path | None
) -> list[tuple[dict, Artwork | None]]:
    """Each `batch-classify` result row with the artwork it matched, if known.

    Rows carry their artwork already when `batch-classify` had the scrape. The
    scrape at `artworks` is only read for rows without one.
    """
    known = None
    if artworks is not None and any(not row.get("artwork") for row in rows):
        known = index(artworks)
    return [
        (
            row,
            Artwork(**row["artwork"])
            if row.get("artwork")
            else artwork_of(row.get("artwork_path", ""), known),
        )
        for row in rows
    ]


async def complete(
    results: Path,
    *,
    output: Path,
    jsonl: Path,
    artworks: Path | None = None,
    everything: bool = False,
    concurrency: int = 10,
    rate: float = 5.0,
    report: Reporter | None = None,
) -> None:
    """Fill in the missing fields of the artworks a `batch-classify` run matched.

    Completed artworks are appended to `jsonl`, and artworks already in it are not
    fetched again, so an interrupted run can simply be started again. `output` is
    `results` with every row's artwork replaced by its completed record.
    """
    with open_text(results) as file:
        data = json.load(file)
    pairs = matched_artworks(data.get("results", []), artworks)

    wanted = {
        artwork.url: artwork
        for row, artwork in pairs
        if artwork is not None and (everything or row.get("is_match"))
    }
    done = {artwork.url: artwork for artwork in read(jsonl)} if jsonl.exists() else {}
    todo: dict[str, list[Artwork]] = {}
    for url, artwork in wanted.items():
        if url not in done:
            todo.setdefault(artwork.source, []).append(artwork)
    logger.info(
        "%d artworks to complete, %d already in %s",
        sum(map(len, todo.values())),
        len(done),
        jsonl,
    )

    backends = registry()
    skipped = [
        source
        for source in todo
        if source not in backends or not completes(backends[source])
    ]
    if skipped:
        logger.warning("cannot complete artworks from: %s", ", ".join(skipped))

    with jsonl.open("a") as sink:

        async def run(name: str, batch: list[Artwork]) -> None:
            ctx = Context(
                client=client,
                throttle=Throttle(concurrency, rate),
                stats=Stats(workers=concurrency),
                workers=concurrency,
            )
            await drain(name, backends[name].complete(ctx, batch), ctx, sink, report)

        async with make_client(concurrency) as client:
            await asyncio.gather(
                *(
                    run(name, batch)
                    for name, batch in todo.items()
                    if name not in skipped
                )
            )

    completed = {artwork.url: artwork for artwork in read(jsonl)}
    for row, artwork in pairs:
        if artwork is not None:
            row["artwork"] = asdict(completed.get(artwork.url, artwork))
    compressed = output.suffix.lower() == ".zst"
    with open_text(output, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=None if compressed else 2)
    logger.info("wrote %s", output)
