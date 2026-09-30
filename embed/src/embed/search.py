"""Similarity search over embedding matrices that may not fit in memory.

`place` decides where a matrix lives for searching, and says so: on the GPU as
float16 when it fits in free VRAM, otherwise where it already is, which for a
store is a memory map the OS pages in as needed. `topk` then goes over it in
chunks, all queries at once per chunk, so each search reads the matrix once
however many queries there are, and never needs a full copy of it.

Stored embeddings are L2-normalized by every embedder, so the dot product is the
cosine similarity; only the queries are normalized here.
"""

from __future__ import annotations

import logging
import os

import torch

from .store import format_bytes

REF_CHUNK = 262_144
"""Reference rows per step."""
SIMILARITY_BUDGET = 1 << 26
"""Similarities computed at once (256 MB as float32), which sets the query chunk."""
VRAM_MARGIN = 1.2
"""How much more free VRAM than the matrix needs, before it is moved there."""

logger = logging.getLogger(__name__)


def available_ram() -> int | None:
    """What the OS reports as available: free memory plus what it can reclaim."""
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def describe(matrix: torch.Tensor, label: str) -> str:
    rows, dim = matrix.shape
    dtype = str(matrix.dtype).removeprefix("torch.")
    return f"{label}: {rows:,} x {dim} {dtype} = {format_bytes(nbytes(matrix))}"


def nbytes(matrix: torch.Tensor, dtype: torch.dtype | None = None) -> int:
    itemsize = torch.empty((), dtype=dtype or matrix.dtype).element_size()
    return matrix.nelement() * itemsize


def place(
    matrix: torch.Tensor,
    device: str,
    *,
    label: str = "embeddings",
    scans_per_query: bool = False,
) -> torch.Tensor:
    """Move `matrix` to where searching it is fastest, and report where that is.

    Always logs the size, whether it went to VRAM, and whether it fits in RAM,
    since that decides whether every full scan reads the file from disk again.
    `scans_per_query` is for callers that search once per incoming query, where a
    matrix that does not fit in RAM costs one full read per query.
    """
    logger.info(describe(matrix, label))
    on_gpu, lines = placement(
        nbytes(matrix), nbytes(matrix, torch.float16), matrix.dtype, device
    )
    for level, line in lines:
        if level == logging.WARNING and scans_per_query:
            line += ", and every text or upload query is one full scan"
        logger.log(level, line)
    return matrix.to(device=device, dtype=torch.float16) if on_gpu else matrix


def placement(
    size: int, gpu_size: int, dtype: torch.dtype, device: str
) -> tuple[bool, list[tuple[int, str]]]:
    """Whether a matrix goes to the GPU, and the report lines that explain it.

    Needs only sizes, so `embed info` can say where a store would go on this
    machine without loading it.
    """
    lines: list[tuple[int, str]] = []
    if device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if gpu_size * VRAM_MARGIN <= free:
            return True, [
                (
                    logging.INFO,
                    f"VRAM: on {device} ({format_bytes(gpu_size)} of "
                    f"{format_bytes(free)} free)",
                ),
                (logging.INFO, "RAM:  not needed for search (on GPU)"),
                (logging.INFO, f"compute: float16 on {device}"),
            ]
        lines.append(
            (
                logging.INFO,
                f"VRAM: not used, needs {format_bytes(gpu_size * VRAM_MARGIN)}, "
                f"{format_bytes(free)} free on {device}",
            )
        )
    else:
        lines.append((logging.INFO, "VRAM: not used (no CUDA device)"))

    available = available_ram()
    if available is None:
        lines.append(
            (
                logging.INFO,
                f"RAM:  unknown how much is available, needs {format_bytes(size)}",
            )
        )
    elif size <= available:
        lines.append(
            (
                logging.INFO,
                f"RAM:  fits ({format_bytes(size)} of {format_bytes(available)} "
                "available), repeated searches read from RAM",
            )
        )
    else:
        lines.append(
            (
                logging.WARNING,
                f"RAM:  does NOT fit ({format_bytes(size)}, {format_bytes(available)} "
                "available), every full scan reads the file from disk again",
            )
        )

    converted = " (each chunk converted from float16)" if dtype == torch.float16 else ""
    lines.append((logging.INFO, f"compute: float32 on cpu{converted}"))
    return False, lines


def topk(
    queries: torch.Tensor, refs: torch.Tensor, k: int, *, chunk: int = REF_CHUNK
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `k` most similar refs for every query, as `(similarities, indices)`.

    One pass over `refs`, a chunk of rows at a time and every query against each
    chunk, keeping a running top-k. Works on whatever device `refs` is on: float16
    on a GPU, float32 on the CPU (a float16 chunk is converted as it is read). The
    results are float32 and int64 on the CPU, shaped `(len(queries), k)`.
    """
    device = refs.device
    compute = torch.float16 if device.type == "cuda" else torch.float32
    queries = queries.to(device=device, dtype=compute)
    queries = queries / queries.norm(dim=-1, keepdim=True)
    k = min(k, len(refs))
    step = max(1, SIMILARITY_BUDGET // chunk)

    values = torch.full((len(queries), k), -torch.inf, dtype=compute, device=device)
    indices = torch.zeros((len(queries), k), dtype=torch.long, device=device)
    for start in range(0, len(refs), chunk):
        block = refs[start : start + chunk].to(device=device, dtype=compute)
        for first in range(0, len(queries), step):
            rows = slice(first, first + step)
            similarities = queries[rows] @ block.T
            found, where = torch.topk(similarities, min(k, len(block)), dim=1)
            merged, pick = torch.topk(torch.cat([values[rows], found], dim=1), k, dim=1)
            indices[rows] = torch.gather(
                torch.cat([indices[rows], where + start], dim=1), 1, pick
            )
            values[rows] = merged

    return values.float().cpu(), indices.cpu()


def perform_search(top_k, query_embedding, image_embeddings, image_paths):
    """The `top_k` images most similar to one query, as `(indices, similarities)`."""
    values, indices = topk(query_embedding.reshape(1, -1), image_embeddings, top_k)
    return indices[0], values[0]


def find_similar(query_index, top_k, image_embeddings, image_paths, exclude_self=True):
    """Images similar to the one at `query_index`, as `(indices, similarities)`."""
    query = image_embeddings[query_index : query_index + 1]
    values, indices = topk(query, image_embeddings, top_k + exclude_self)
    keep = indices[0] != query_index if exclude_self else slice(None)
    return indices[0][keep][:top_k], values[0][keep][:top_k]
