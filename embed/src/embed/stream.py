"""Embedding references that may be URLs into a store, fetching while the model works.

Reading the images is IO bound and embedding them is not, so the two are run at
the same time: fetches stay in flight while the current batch is on the GPU, and
no image is ever written to disk. Each batch goes into the store as soon as it is
embedded, so an interrupted run keeps its work and the next one carries on.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from common.images import reason, stream

from .models import EmbeddingModel
from .store import Store

if TYPE_CHECKING:
    import torch
    from PIL import Image

logger = logging.getLogger(__name__)

HASH_FLUSH = 1024
"""Hashes recorded without embedding are written in groups of this many."""


async def encode_stream(
    embedder: EmbeddingModel,
    refs: Iterable[str],
    *,
    batch_size: int = 64,
    keep: Callable[[str, str], bool] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
    concurrency: int | None = None,
    per_host: int | None = None,
    rate: float | None = None,
) -> AsyncIterator[tuple[list[str], list[str], torch.Tensor]]:
    """Embed every reference, yielding `(refs, digests, embeddings)` a batch at a time.

    `keep(ref, digest)` is asked for every image once it is loaded, before it is
    embedded, so a known and unchanged image costs no GPU time. `on_error` gets
    every image that could not be read. The refs are the original strings (a URL
    stays a URL) and only cover images that loaded. Order follows completion, not
    input.

    Up to a full batch of loaded images is buffered, so the next batch is being
    fetched while the model works on this one and is ready when it is done. That
    also means up to two batches of decoded images are in memory at once.
    """
    batch: list[tuple[str, Image.Image, str]] = []
    fetched = stream(
        refs,
        concurrency=concurrency,
        per_host=per_host,
        rate=rate,
        buffer=batch_size,
        on_error=on_error,
    )
    async for ref, image, digest in fetched:
        if keep is not None and not keep(ref, digest):
            continue
        batch.append((ref, image, digest))
        if len(batch) >= batch_size:
            # the model call blocks, so let the fetchers keep working meanwhile
            yield await asyncio.to_thread(_encode, embedder, batch)
            batch = []
    if batch:
        yield await asyncio.to_thread(_encode, embedder, batch)


def _encode(
    embedder: EmbeddingModel, batch: list[tuple[str, Image.Image, str]]
) -> tuple[list[str], list[str], torch.Tensor]:
    refs, images, digests = (list(column) for column in zip(*batch, strict=True))
    return refs, digests, embedder.encode_pil_batch(images).cpu()


def pending(
    store: Store, refs: Mapping[str, str], *, verify: bool
) -> tuple[list[str], int]:
    """The images a run has to read, and how many were left out as failed before.

    That is the ones missing from the store, except those that failed for a lasting
    reason on an earlier run (a 404, a broken file). With `verify` it is all of them.
    """
    if verify:
        return list(refs), 0
    stored = set(store.refs())
    failed = store.failures()
    missing = [(source, key) for source, key in refs.items() if key not in stored]
    todo = [source for source, key in missing if key not in failed]
    return todo, len(missing) - len(todo)


@dataclass
class Progress:
    total: int
    loaded: int = 0
    embedded: int = 0
    hashed: int = 0
    """Stored rows that had no hash yet: hashed, not embedded again."""
    failed: Counter[str] = field(default_factory=Counter)
    """Images that could not be read, by reason."""

    @property
    def done(self) -> int:
        return self.loaded + self.failed.total()


def encode_into(
    store: Store,
    embedder: EmbeddingModel,
    refs: Mapping[str, str],
    todo: list[str],
    *,
    batch_size: int = 64,
    report: Callable[[Progress], None] | None = None,
    concurrency: int | None = None,
    per_host: int | None = None,
    rate: float | None = None,
) -> Progress:
    """Embed `todo` into `store`, skipping images it already holds unchanged.

    `refs` maps where each image is read from (a local path or a URL) to its key
    in the store (the path relative to the store's root, or the same URL). An image
    whose key is stored with the same hash is skipped; one stored without a hash,
    from before hashes were kept, gets its hash recorded and keeps its vector.

    Images that fail for a lasting reason are recorded in the store, so the next
    run skips them; one that loads again is taken off that list.
    """
    known = store.hashes()
    recorded = store.failures()
    before = dict(recorded)
    progress = Progress(total=len(todo))
    unhashed: dict[str, str] = {}

    def flush() -> None:
        store.put_hashes(unhashed)
        unhashed.clear()

    def failed(source: str, error: Exception) -> None:
        why, lasting = reason(error)
        progress.failed[why] += 1
        logger.debug(f"skipping {source}: {error}")
        if lasting:
            recorded[refs[source]] = why
        else:
            recorded.pop(refs[source], None)
        if report is not None:
            report(progress)

    def keep(source: str, digest: str) -> bool:
        progress.loaded += 1
        if report is not None:
            report(progress)
        key = refs[source]
        recorded.pop(key, None)
        if key not in known:
            return True
        if known[key] is None:
            unhashed[key] = digest
            progress.hashed += 1
            if len(unhashed) >= HASH_FLUSH:
                flush()
            return False
        return known[key] != digest

    async def run() -> None:
        batches = encode_stream(
            embedder,
            todo,
            batch_size=batch_size,
            keep=keep,
            on_error=failed,
            concurrency=concurrency,
            per_host=per_host,
            rate=rate,
        )
        async for sources, digests, embeddings in batches:
            store.put([refs[s] for s in sources], digests, embeddings.numpy())
            progress.embedded += len(sources)
            if report is not None:
                report(progress)

    try:
        asyncio.run(run())
    finally:
        flush()
        if recorded != before:
            store.set_failures(recorded)
    return progress
