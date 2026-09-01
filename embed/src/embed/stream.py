"""Embedding references that may be URLs, fetching while the model works.

Reading the images is IO bound and embedding them is not, so the two are run at
the same time: fetches stay in flight while the current batch is on the GPU, and
no image is ever written to disk.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from common.images import stream

from .embed import EmbeddingModel

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


async def encode_stream(
    embedder: EmbeddingModel,
    refs: Iterable[str],
    *,
    batch_size: int = 64,
    concurrency: int | None = None,
    on_batch=None,
) -> tuple[torch.Tensor, list[str]]:
    """Embed every reference, returning the embeddings and the refs they belong to.

    The returned refs are the identity used everywhere downstream, so they are the
    original strings (a URL stays a URL) and only cover images that actually
    loaded. Order follows completion, not input.
    """
    embeddings: list[torch.Tensor] = []
    done: list[str] = []
    batch_refs: list[str] = []
    batch_images = []

    def flush() -> None:
        if not batch_images:
            return
        embeddings.append(embedder.encode_pil_batch(batch_images).cpu())
        done.extend(batch_refs)
        if on_batch is not None:
            on_batch(len(done))
        batch_refs.clear()
        batch_images.clear()

    async for ref, image in stream(refs, concurrency=concurrency):
        batch_refs.append(ref)
        batch_images.append(image)
        if len(batch_images) >= batch_size:
            # the model call blocks, so let the fetchers keep working meanwhile
            await asyncio.to_thread(flush)

    await asyncio.to_thread(flush)

    if not embeddings:
        raise ValueError("no images could be loaded")

    import torch

    return torch.cat(embeddings), done


def encode_refs(
    embedder: EmbeddingModel,
    refs: Iterable[str],
    *,
    batch_size: int = 64,
    concurrency: int | None = None,
    on_batch=None,
) -> tuple[torch.Tensor, list[str]]:
    """Blocking wrapper around `encode_stream`, for use from the CLI."""
    return asyncio.run(
        encode_stream(
            embedder,
            refs,
            batch_size=batch_size,
            concurrency=concurrency,
            on_batch=on_batch,
        )
    )
