"""`unstir match`: retrieval, matching into a match store, and a decision per plate.

1. Plates: the plates of the chosen clusters of a clustering result, or every
   plate in the plate embeddings when no clusters are chosen.
2. Retrieval: the `top_k` nearest references of each plate, by embedding.
3. Matching: the candidate pairs the store has no score for yet get the scores of
   `match.geometry`, committed to the store per batch of plates.
4. Decision: `decide.run` on every candidate pair of this run.
"""

import logging
import os
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import numpy as np

from common.errors import UserError
from common.paths import rebase_paths
from common.progress import progress

from .decide import Rule
from .match import Matcher, Settings, geometry, prepare
from .store import MatchStore, relative

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
    keys: tuple[list[str], list[str]],
    settings: Settings,
    device: str,
    batch: int,
) -> Iterator[list[dict]]:
    """The scores of `todo`, one list per `batch` plates. References are
    extracted once and kept on the device; plates are read ahead on threads
    while the GPU works. Rows hold the refs of `keys` (plates, references)."""
    matcher = Matcher(settings, device)
    by_plate: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in todo:
        by_plate[candidate.plate].append(candidate)
    read = readable(settings.resize)
    plate_keys, reference_keys = keys

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

        def collect(futures: list[tuple[Candidate, Future]]) -> list[dict]:
            return [
                {
                    "plate": plate_keys[c.plate],
                    "reference": reference_keys[c.reference],
                    **future.result(),
                }
                for c, future in futures
            ]

        task = bar.add_task("plates", total=len(by_plate))
        futures: list[tuple[Candidate, Future]] = []
        plates_in_batch = 0
        for row, image in prefetch(io, lambda r: read(plate_refs[r]), by_plate):
            bar.advance(task)
            if image is not None:
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
            plates_in_batch += 1
            if plates_in_batch == batch:
                yield collect(futures)
                futures, plates_in_batch = [], 0
        yield collect(futures)


@dataclass(frozen=True)
class Options:
    store: Path
    clusters: Path | None
    cluster_ids: frozenset[int]
    """Empty: every plate in the plate embeddings."""
    plate_embeddings: Path
    plate_images: Path
    reference_embeddings: Path
    reference_images: Path
    top_k: int
    plates: int
    """Use at most this many of the plates, at random, 0 for all."""
    batch: int
    seed: int
    rescore: bool
    settings: Settings
    rule: Rule


def run(options: Options) -> None:
    from embed.device import get_device
    from embed.io import load_embeddings
    from embed.search import place, topk

    from . import decide

    start = time.perf_counter()
    device = get_device()
    store = MatchStore.create_or_open(
        options.store, options.settings, rescore=options.rescore
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

    if options.cluster_ids:
        if options.clusters is None:
            raise UserError("--cluster needs --clusters, or paths.clusters")
        wanted = cluster_plates(
            options.clusters, set(options.cluster_ids), options.plate_images
        )
        plates = [row for row, ref in enumerate(plate_refs) if ref in wanted]
        if len(plates) < len(wanted):
            logger.warning(
                f"{len(wanted) - len(plates):,} plates of the clusters are not in "
                f"{options.plate_embeddings}"
            )
    else:
        plates = list(range(len(plate_refs)))
    if options.plates and options.plates < len(plates):
        plates = sorted(random.Random(options.seed).sample(plates, options.plates))
    if not plates:
        raise UserError("no plates to match")

    references = place(reference_store["embeddings"], device, label="references")
    similarities, indices = topk(
        plate_store["embeddings"][plates], references, options.top_k
    )
    del references
    similarities, indices = similarities.numpy(), indices.numpy()
    candidates = [
        Candidate(plate, int(indices[i, rank]), rank, float(similarities[i, rank]))
        for i, plate in enumerate(plates)
        for rank in range(indices.shape[1])
    ]

    plate_keys = [relative(ref, options.plate_images) for ref in plate_refs]
    reference_keys = [relative(ref, options.reference_images) for ref in reference_refs]
    done = store.scored()
    todo = [
        c
        for c in candidates
        if (plate_keys[c.plate], reference_keys[c.reference]) not in done
    ]
    logger.info(
        f"{len(plates):,} plates, {len(reference_refs):,} references, "
        f"{len(candidates):,} candidate pairs, {len(candidates) - len(todo):,} "
        f"already in {options.store}, {len(todo):,} to match on {device}"
    )

    if todo:
        for rows in score(
            todo,
            plate_refs,
            reference_refs,
            (plate_keys, reference_keys),
            options.settings,
            device,
            options.batch,
        ):
            store.add(rows)

    store.set_candidates(
        [
            {
                "plate": plate_keys[c.plate],
                "reference": reference_keys[c.reference],
                "rank": c.rank,
                "similarity": c.similarity,
            }
            for c in candidates
        ]
    )
    store.update_meta(
        plate_embeddings=str(options.plate_embeddings),
        reference_embeddings=str(options.reference_embeddings),
        model=plate_store["model_id"],
        clusters=str(options.clusters) if options.cluster_ids else None,
        cluster_ids=sorted(options.cluster_ids),
        top_k=options.top_k,
        plates=len(plates),
        references=len(reference_refs),
        matched_last_run=len(todo),
        seconds_last_run=round(time.perf_counter() - start),
        updated=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    decide.run(store, options.rule, options.plate_images, options.reference_images)
