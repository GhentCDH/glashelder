"""Candidate pairs to label: retrieval, matching, and a sample by score.

1. Plates: the plates in the chosen clusters of a clustering result.
2. Retrieval: the `top_k` nearest references of each plate, by embedding.
3. Matching: every candidate pair gets the scores of `match.geometry`.
4. Sample: pairs spread evenly over the `cells` values, so the labels cover the
   band where match and no match mix, not only the many pairs that score 0.

A few `outside` pairs come from below the top k (ranks `top_k` to
`top_k * OUTSIDE_FACTOR`). A label of match on one of them shows that retrieval
misses true pairs, which no rule after it can fix.

The output directory holds `scores.parquet` (every scored pair), `pairs.jsonl`
(the sample, for `unstir serve`) and `meta.json` (the settings).
"""

import json
import logging
import os
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice, zip_longest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common.errors import UserError
from common.fs import atomic_write
from common.paths import rebase_paths
from common.progress import progress

from .match import Matcher, Settings, geometry, prepare

OUTSIDE_FACTOR = 10
IO_THREADS = 16
PREFETCH = 64

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    plate: int
    """Row in the plate store."""
    reference: int
    """Row in the reference store."""
    rank: int
    """0 is the nearest reference of the plate."""
    similarity: float
    outside: bool


def cluster_plates(clusters: Path, ids: set[int], plate_images: Path) -> set[str]:
    """Paths of the plates in the clusters `ids`, under `plate_images`."""
    from cluster import result

    clustering = result.load(clusters)
    unknown = ids - set(clustering.labels.tolist())
    if unknown:
        raise UserError(f"{clusters} has no clusters {sorted(unknown)}")
    refs = rebase_paths(clustering.refs, plate_images)
    keep = np.isin(clustering.labels, list(ids))
    return {ref for ref, chosen in zip(refs, keep.tolist(), strict=True) if chosen}


def candidates(
    plates: list[int],
    similarities: np.ndarray,
    indices: np.ndarray,
    top_k: int,
    outside: int,
    rng: random.Random,
) -> list[Candidate]:
    """The top k of every plate, and `outside` pairs from the ranks after it."""
    inside = [
        Candidate(
            plate, int(indices[i, rank]), rank, float(similarities[i, rank]), False
        )
        for i, plate in enumerate(plates)
        for rank in range(min(top_k, indices.shape[1]))
    ]
    if indices.shape[1] <= top_k:
        return inside
    far = [
        (i, rng.randrange(top_k, indices.shape[1]))
        for i in rng.choices(range(len(plates)), k=outside)
    ]
    return inside + [
        Candidate(
            plates[i], int(indices[i, rank]), rank, float(similarities[i, rank]), True
        )
        for i, rank in dict.fromkeys(far)
    ]


def prefetch[T, R](
    pool: ThreadPoolExecutor, work: Callable[[T], R], items: Iterable[T]
) -> Iterator[tuple[T, R]]:
    """`(item, work(item))` in order, with up to `PREFETCH` items in progress."""
    items = iter(items)
    pending = deque((item, pool.submit(work, item)) for item in islice(items, PREFETCH))
    while pending:
        item, future = pending.popleft()
        pending.extend((more, pool.submit(work, more)) for more in islice(items, 1))
        yield item, future.result()


def readable(resize: int) -> Callable[[str], np.ndarray | None]:
    def read(ref: str) -> np.ndarray | None:
        try:
            return prepare(ref, resize)
        except OSError as error:
            logger.warning(f"could not read {ref}: {error}")
            return None

    return read


def score(
    todo: list[Candidate],
    plate_refs: list[str],
    reference_refs: list[str],
    settings: Settings,
    device: str,
) -> list[dict]:
    """The scores of every candidate pair. References are extracted once and kept
    on the device; plates are read ahead on threads while the GPU works."""
    matcher = Matcher(settings, device)
    by_plate: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in todo:
        by_plate[candidate.plate].append(candidate)
    read = readable(settings.resize)

    with (
        ThreadPoolExecutor(IO_THREADS) as io,
        ThreadPoolExecutor(os.cpu_count() or 8) as cpu,
        progress() as bar,
    ):
        needed = sorted({c.reference for c in todo})
        task = bar.add_task("references", total=len(needed))
        references = {}
        for row, image in prefetch(io, lambda r: read(reference_refs[r]), needed):
            if image is not None:
                references[row] = (matcher.extract(image), image.shape)
            bar.advance(task)

        task = bar.add_task("plates", total=len(by_plate))
        futures: list[tuple[Candidate, Future]] = []
        for row, image in prefetch(io, lambda r: read(plate_refs[r]), by_plate):
            bar.advance(task)
            if image is None:
                continue
            features = matcher.extract(image)
            for candidate in by_plate[row]:
                if candidate.reference not in references:
                    continue
                theirs, shape = references[candidate.reference]
                pts_plate, pts_reference = matcher.match(features, theirs)
                futures.append(
                    (
                        candidate,
                        cpu.submit(
                            geometry,
                            pts_plate,
                            pts_reference,
                            image.shape,
                            shape,
                            settings,
                        ),
                    )
                )

        return [
            {
                "plate": plate_refs[c.plate],
                "reference": reference_refs[c.reference],
                "rank": c.rank,
                "similarity": c.similarity,
                "outside": c.outside,
                **future.result(),
            }
            for c, future in futures
        ]


def sample(
    rows: list[dict],
    n: int,
    rng: random.Random,
    key: Callable[[dict], int] = lambda row: row["cells"],
) -> list[dict]:
    """`n` pairs, taken in turn from every value of `key`, highest first."""
    groups_by: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups_by[key(row)].append(row)
    groups = [rng.sample(g, len(g)) for _, g in sorted(groups_by.items(), reverse=True)]
    turns = (row for turn in zip_longest(*groups) for row in turn if row is not None)
    return list(islice(turns, n))


@dataclass(frozen=True)
class Options:
    clusters: Path
    cluster_ids: frozenset[int]
    plate_embeddings: Path
    plate_images: Path
    reference_embeddings: Path
    reference_images: Path
    top_k: int
    plates: int
    """Use at most this many plates of the clusters, 0 for all."""
    sample: int
    outside: int
    seed: int
    settings: Settings


def run(options: Options, output: Path) -> None:
    from embed.device import get_device
    from embed.io import load_embeddings
    from embed.search import place, topk

    start = time.perf_counter()
    rng = random.Random(options.seed)
    device = get_device()

    wanted = cluster_plates(
        options.clusters, set(options.cluster_ids), options.plate_images
    )
    plate_store = load_embeddings(options.plate_embeddings, root=options.plate_images)
    reference_store = load_embeddings(
        options.reference_embeddings, root=options.reference_images
    )
    if plate_store["model_id"] != reference_store["model_id"]:
        raise UserError(
            f"the plates are embedded with {plate_store['model_id']} and the "
            f"references with {reference_store['model_id']}"
        )
    plate_refs: list[str] = plate_store["images"]
    reference_refs: list[str] = reference_store["images"]

    plates = [row for row, ref in enumerate(plate_refs) if ref in wanted]
    if len(plates) < len(wanted):
        logger.warning(
            f"{len(wanted) - len(plates):,} plates of the clusters are not in "
            f"{options.plate_embeddings}"
        )
    if options.plates and options.plates < len(plates):
        plates = sorted(rng.sample(plates, options.plates))
    if not plates:
        raise UserError("no plates to match")

    references = place(reference_store["embeddings"], device, label="references")
    similarities, indices = topk(
        plate_store["embeddings"][plates],
        references,
        options.top_k * OUTSIDE_FACTOR,
    )
    del references
    todo = candidates(
        plates,
        similarities.numpy(),
        indices.numpy(),
        options.top_k,
        options.outside,
        rng,
    )
    logger.info(
        f"{len(plates):,} plates, {len(reference_refs):,} references, "
        f"{len(todo):,} pairs to match on {device}"
    )

    rows = score(todo, plate_refs, reference_refs, options.settings, device)
    inside = [row for row in rows if not row["outside"]]
    chosen = sample(inside, options.sample, rng) + [r for r in rows if r["outside"]]

    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output / "scores.parquet")
    atomic_write(output / "pairs.jsonl", "".join(json.dumps(r) + "\n" for r in chosen))
    meta = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(options).items()},
        "cluster_ids": sorted(options.cluster_ids),
        "plates_used": len(plates),
        "pairs_scored": len(rows),
        "pairs_sampled": len(chosen),
        "model": plate_store["model_id"],
        "seconds": round(time.perf_counter() - start),
    }
    atomic_write(output / "meta.json", json.dumps(meta, indent=2) + "\n")

    levels = np.bincount([r["cells"] for r in inside])
    logger.info(
        f"{len(rows):,} pairs scored in {meta['seconds']}s, {len(chosen):,} in "
        f"{output / 'pairs.jsonl'}. Pairs per cells value: "
        + ", ".join(f"{c}: {n}" for c, n in enumerate(levels.tolist()) if n)
    )
