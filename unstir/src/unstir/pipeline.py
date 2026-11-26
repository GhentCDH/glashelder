"""`unstir match`: retrieval, matching into a match store, and a decision per plate.

1. Plates: the plates of the chosen clusters and cluster groups of a clustering
   result, or every plate in the plate embeddings when none are chosen.
2. Retrieval: the `top_k` nearest references of each plate, by embedding.
3. Matching: the candidate pairs the store has no score for yet get the scores of
   `match.geometry`, committed to the store every `batch` references.
4. Decision: `decide.run` on every candidate pair of this run.

Matching goes reference by reference, so the number of references does not
decide the memory. The plates of a chunk are extracted first and kept on the
GPU in float16 (about 0.5 MB each). Then every reference that is a candidate of
one of those plates is read, extracted, matched with all of them, and dropped.
Every image is extracted once per chunk; with all plates in one chunk, once.
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
from itertools import batched, islice
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


@dataclass(frozen=True)
class Selection:
    """Which plates of a clustering to match."""

    clusters: Path | None
    ids: frozenset[int] = frozenset()
    groups: frozenset[str] = frozenset()
    """Cluster groups from the clustering's `annotations.json`."""
    without: frozenset[str] = frozenset()
    """Groups whose clusters are left out, also when an id or group names them."""
    soft: bool = False
    """Also the noise plates whose strongest soft cluster is chosen."""

    @property
    def chosen(self) -> bool:
        return bool(self.ids or self.groups)


def select_plates(selection: Selection, plate_images: Path) -> set[str]:
    """Paths of the chosen plates, under `plate_images`."""
    from cluster import result

    if selection.clusters is None:
        raise UserError("choosing clusters needs --clusters, or paths.clusters")
    clustering = result.load(selection.clusters)
    groups = result.read_annotations(selection.clusters).groups
    if unknown := (selection.groups | selection.without) - set(groups):
        raise UserError(
            f"{selection.clusters} has no groups {', '.join(sorted(unknown))}. "
            f"It has: {', '.join(sorted(groups))}"
        )

    def members(names: frozenset[str]) -> set[int]:
        return {i for name in names for i in groups[name]}

    ids = (set(selection.ids) | members(selection.groups)) - members(selection.without)
    if unknown := set(selection.ids) - set(clustering.labels.tolist()):
        raise UserError(f"{selection.clusters} has no clusters {sorted(unknown)}")

    keep = np.isin(clustering.labels, list(ids))
    in_clusters = int(keep.sum())
    if selection.soft:
        if clustering.soft_labels is None:
            raise UserError(f"{selection.clusters} has no soft clustering")
        keep |= (clustering.labels == -1) & np.isin(
            clustering.soft_labels[:, 0], list(ids)
        )
    logger.info(
        f"{len(ids)} clusters: {in_clusters:,} plates"
        + (
            f", and {int(keep.sum()) - in_clusters:,} noise plates"
            if selection.soft
            else ""
        )
    )
    refs = rebase_paths(clustering.refs, plate_images)
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


@dataclass(frozen=True)
class Work:
    """What `score` needs besides the candidates."""

    plate_refs: list[str]
    reference_refs: list[str]
    plate_keys: list[str]
    """The plate refs as the store keeps them."""
    reference_keys: list[str]
    settings: Settings
    device: str
    batch: int
    """References per list of rows."""
    chunk: int
    """Plates kept on the GPU at once."""


def score(todo: list[Candidate], work: Work) -> Iterator[list[dict]]:
    """The scores of `todo`, one list of rows per `work.batch` references."""
    matcher = Matcher(work.settings, work.device)
    read = readable(work.settings.resize)
    plates = sorted({c.plate for c in todo})
    chunks = list(batched(plates, work.chunk, strict=False))

    with (
        ThreadPoolExecutor(IO_THREADS) as io,
        ThreadPoolExecutor(os.cpu_count() or 8) as cpu,
        progress() as bar,
    ):

        def collect(futures: list[tuple[Candidate, Future]]) -> list[dict]:
            return [
                {
                    "plate": work.plate_keys[c.plate],
                    "reference": work.reference_keys[c.reference],
                    **future.result(),
                }
                for c, future in futures
            ]

        for number, chunk in enumerate(chunks, 1):
            label = f" {number}/{len(chunks)}" if len(chunks) > 1 else ""
            members = set(chunk)
            task = bar.add_task(f"plates{label}", total=len(chunk))
            kept = {}
            for row, image in prefetch(io, lambda r: read(work.plate_refs[r]), chunk):
                if image is not None:
                    kept[row] = (Matcher.compact(matcher.extract(image)), image.shape)
                bar.advance(task)

            by_reference: dict[int, list[Candidate]] = defaultdict(list)
            for c in todo:
                if c.plate in members and c.plate in kept:
                    by_reference[c.reference].append(c)
            task = bar.add_task(f"references{label}", total=len(by_reference))
            futures: list[tuple[Candidate, Future]] = []
            done = 0
            for row, image in prefetch(
                io, lambda r: read(work.reference_refs[r]), sorted(by_reference)
            ):
                bar.advance(task)
                if image is not None:
                    theirs = matcher.extract(image)
                    for c in by_reference[row]:
                        features, shape = kept[c.plate]
                        pts_plate, pts_reference = matcher.match(
                            Matcher.expand(features), theirs
                        )
                        futures.append(
                            (
                                c,
                                cpu.submit(
                                    geometry,
                                    pts_plate,
                                    pts_reference,
                                    shape,
                                    image.shape,
                                    work.settings,
                                ),
                            )
                        )
                done += 1
                if done == work.batch:
                    yield collect(futures)
                    futures, done = [], 0
            yield collect(futures)
            del kept


@dataclass(frozen=True)
class Options:
    store: Path
    selection: Selection
    plate_embeddings: Path
    plate_images: Path
    reference_embeddings: Path
    reference_images: Path
    top_k: int
    plates: int
    """Use at most this many of the plates, at random, 0 for all."""
    batch: int
    chunk: int
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

    selection = options.selection
    if selection.chosen:
        wanted = select_plates(selection, options.plate_images)
        plates = [row for row, ref in enumerate(plate_refs) if ref in wanted]
        if len(plates) < len(wanted):
            logger.warning(
                f"{len(wanted) - len(plates):,} chosen plates are not in "
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
        work = Work(
            plate_refs=plate_refs,
            reference_refs=reference_refs,
            plate_keys=plate_keys,
            reference_keys=reference_keys,
            settings=options.settings,
            device=device,
            batch=options.batch,
            chunk=options.chunk,
        )
        for rows in score(todo, work):
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
        clusters=str(selection.clusters) if selection.chosen else None,
        cluster_ids=sorted(selection.ids),
        groups=sorted(selection.groups),
        without_groups=sorted(selection.without),
        soft=selection.soft,
        top_k=options.top_k,
        plates=len(plates),
        references=len(reference_refs),
        matched_last_run=len(todo),
        seconds_last_run=round(time.perf_counter() - start),
        updated=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    decide.run(store, options.rule, options.plate_images, options.reference_images)
