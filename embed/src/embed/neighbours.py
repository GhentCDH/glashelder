"""The exact top-k references for a set of query images, saved and kept up to date.

Searching every glashelder image against a million references is minutes of CPU
work, and serving needs all of it before the first request. But the answer only
changes when one of the two stores does, and stores mostly grow at the end:

- new reference rows can push into a query's top-k, but do not change how the
  old rows rank, so searching only the new rows and merging is exact;
- new query rows need a search of their own, against all references.

So the result is saved inside the reference store, in `neighbours/`, one file per
query store, and each use searches only what is new since. Anything else (rows
changed in place or removed, which a store counts as `edits`, another model, a
different set of queries) starts over from scratch.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from common.fs import atomic_writer

from .search import merge, topk
from .store import Store

DIRECTORY = "neighbours"

logger = logging.getLogger(__name__)


@dataclass
class Neighbours:
    similarities: np.ndarray
    """`(queries, k)` float32, best first."""
    indices: np.ndarray
    """`(queries, k)` int64: rows of the reference store."""
    rows: int
    """Reference rows searched."""
    edits: int
    """The reference store's `edits` when they were searched."""
    queries: int
    """Query rows searched."""
    queries_hash: str
    """Fingerprint of those query refs and the query store's `edits`."""
    model_id: str


def queries_hash(refs: Sequence[str], edits: int) -> str:
    digest = hashlib.blake2b(f"{edits}\n".encode(), digest_size=16)
    for ref in refs:
        digest.update(f"{ref}\n".encode())
    return digest.hexdigest()


def path_for(references: Store, queries: Store) -> Path:
    return references.path / DIRECTORY / f"{queries.path.name}.npz"


def load(path: Path) -> Neighbours | None:
    if not path.is_file():
        return None
    with np.load(path) as saved:
        return Neighbours(
            similarities=saved["similarities"],
            indices=saved["indices"],
            rows=int(saved["rows"]),
            edits=int(saved["edits"]),
            queries=int(saved["queries"]),
            queries_hash=str(saved["queries_hash"]),
            model_id=str(saved["model_id"]),
        )


def save(path: Path, neighbours: Neighbours) -> None:
    path.parent.mkdir(exist_ok=True)
    with atomic_writer(path, binary=True) as file:
        np.savez(file, **asdict(neighbours))


def usable(
    saved: Neighbours | None,
    *,
    query_refs: Sequence[str],
    query_edits: int,
    rows: int,
    ref_edits: int,
    model_id: str,
    k: int,
) -> bool:
    """Whether `saved` is still right for what it covers, so it can be extended."""
    return (
        saved is not None
        and saved.model_id == model_id
        and saved.edits == ref_edits
        and saved.indices.shape[1] >= min(k, rows)
        and saved.rows <= rows
        and saved.queries <= len(query_refs)
        and saved.queries_hash == queries_hash(query_refs[: saved.queries], query_edits)
    )


def update(
    saved: Neighbours | None,
    *,
    query_refs: Sequence[str],
    query_edits: int,
    query_vectors: torch.Tensor,
    ref_vectors: torch.Tensor,
    ref_edits: int,
    model_id: str,
    k: int,
) -> tuple[Neighbours, str]:
    """`saved` brought up to date, searching only what is new, and what that took."""
    if not usable(
        saved,
        query_refs=query_refs,
        query_edits=query_edits,
        rows=len(ref_vectors),
        ref_edits=ref_edits,
        model_id=model_id,
        k=k,
    ):
        similarities, indices = topk(
            query_vectors,
            ref_vectors,
            k,
            description=f"top {k} of {len(ref_vectors):,} references "
            f"for {len(query_refs):,} images",
        )
        done = "searched from scratch"
    else:
        assert saved is not None
        similarities = torch.from_numpy(saved.similarities[:, :k])
        indices = torch.from_numpy(saved.indices[:, :k])
        steps = []
        if saved.rows < len(ref_vectors):
            more, where = topk(
                query_vectors[: saved.queries],
                ref_vectors[saved.rows :],
                k,
                description=f"top {k} of {len(ref_vectors) - saved.rows:,} new "
                f"references for {saved.queries:,} images",
            )
            similarities, indices = merge(
                similarities, indices, more, where + saved.rows, k
            )
            steps.append(f"{len(ref_vectors) - saved.rows:,} new references")
        if saved.queries < len(query_refs):
            more, where = topk(
                query_vectors[saved.queries :],
                ref_vectors,
                k,
                description=f"top {k} of {len(ref_vectors):,} references for "
                f"{len(query_refs) - saved.queries:,} new images",
            )
            similarities = torch.cat([similarities, more])
            indices = torch.cat([indices, where])
            steps.append(f"{len(query_refs) - saved.queries:,} new images")
        done = f"searched {' and '.join(steps)}" if steps else "up to date"

    return Neighbours(
        similarities=similarities.numpy(),
        indices=indices.numpy(),
        rows=len(ref_vectors),
        edits=ref_edits,
        queries=len(query_refs),
        queries_hash=queries_hash(query_refs, query_edits),
        model_id=model_id,
    ), done


def neighbours(
    queries: Store,
    references: Store,
    query_vectors: torch.Tensor,
    ref_vectors: torch.Tensor,
    k: int,
    *,
    hint: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The top-k references of every query row, as `(similarities, indices)`.

    Uses and updates what is saved in the reference store. `query_vectors` and
    `ref_vectors` are the stores' matrices, wherever `embed.search.place` put
    them. When the result cannot be saved (no write access), it is still
    returned, and the next use searches again. `hint` is logged before any
    search, for callers that would rather not do that search themselves.
    """
    if queries.model_id != references.model_id:
        raise ValueError(
            f"{queries.path} holds {queries.model_id} embeddings and "
            f"{references.path} {references.model_id}: they cannot be compared"
        )
    path = path_for(references, queries)
    saved = load(path)
    current = (
        usable(
            saved,
            query_refs=queries.refs(),
            query_edits=queries.edits,
            rows=len(ref_vectors),
            ref_edits=references.edits,
            model_id=references.model_id,
            k=k,
        )
        and saved is not None
        and (saved.rows, saved.queries) == (len(ref_vectors), len(queries))
    )
    if not current and hint is not None:
        logger.warning(
            f"the saved top {k} in {path} is missing or not up to date: {hint}"
        )
    result, done = update(
        saved,
        query_refs=queries.refs(),
        query_edits=queries.edits,
        query_vectors=query_vectors,
        ref_vectors=ref_vectors,
        ref_edits=references.edits,
        model_id=references.model_id,
        k=k,
    )
    logger.info(f"top {k} references for {len(queries):,} images: {done}")
    if done != "up to date":
        try:
            save(path, result)
            logger.info(f"saved them in {path}")
        except OSError as e:
            logger.warning(f"could not save {path} ({e}); the next use searches again")
    return result.similarities[:, :k], result.indices[:, :k]
