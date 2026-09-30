"""Embedding references that may be URLs into a store, fetching while the model works.

Reading the images is IO bound and embedding them is not, so the two are run at
the same time: fetches stay in flight while the current batch is on the GPU, and
no image is ever written to disk. Each batch goes into the store as soon as it is
embedded, so an interrupted run keeps its work and the next one carries on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.images import stream

from .embed import EmbeddingModel
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
    concurrency: int | None = None,
    per_host: int | None = None,
    rate: float | None = None,
) -> AsyncIterator[tuple[list[str], list[str], torch.Tensor]]:
    """Embed every reference, yielding `(refs, digests, embeddings)` a batch at a time.

    `keep(ref, digest)` is asked for every image once it is loaded, before it is
    embedded, so a known and unchanged image costs no GPU time. The refs are the
    original strings (a URL stays a URL) and only cover images that loaded. Order
    follows completion, not input.
    """
    batch: list[tuple[str, Image.Image, str]] = []
    fetched = stream(refs, concurrency=concurrency, per_host=per_host, rate=rate)
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


def pending(store: Store, refs: Mapping[str, str], *, verify: bool) -> list[str]:
    """The images a run has to read: the ones missing from the store, or all with `verify`."""
    if verify:
        return list(refs)
    stored = set(store.refs())
    return [source for source, key in refs.items() if key not in stored]


@dataclass
class Progress:
    total: int
    loaded: int = 0
    embedded: int = 0
    hashed: int = 0
    """Stored rows that had no hash yet: hashed, not embedded again."""


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
    """
    known = store.hashes()
    progress = Progress(total=len(todo))
    unhashed: dict[str, str] = {}

    def flush() -> None:
        store.put_hashes(unhashed)
        unhashed.clear()

    def keep(source: str, digest: str) -> bool:
        progress.loaded += 1
        if report is not None:
            report(progress)
        key = refs[source]
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
    return progress
