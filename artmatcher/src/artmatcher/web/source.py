from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from embed import rebase_paths

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from artscraper import Artwork

    from ..loading import MatchingInputs
    from ..ml_classifier import MLClassifier

logger = logging.getLogger(__name__)


@dataclass
class MatchItem:
    glashelder_index: int
    glashelder_path: str
    top_indices: list[int]
    """The top matches by CLIP similarity, best first."""
    similarities: list[float]
    """What the classifier saw for the chosen match (see `batch.shifted`)."""
    is_match: bool
    diff: float
    threshold: float
    classifier_type: str
    features: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    """Which of `top_indices` was chosen; above 0 when earlier ones did not match."""
    top_similarities: list[float] = field(default_factory=list)
    """The CLIP similarities of `top_indices`, in the same order."""

    @property
    def best_index(self) -> int:
        return self.top_indices[self.rank]


class ClassificationSource(ABC):
    """Supplies classification items for the labeling UI."""

    _artworks: dict[str, Artwork]
    """Scraped artwork records, keyed by reference. Set by each subclass."""

    @property
    @abstractmethod
    def total_images(self) -> int: ...

    @property
    def batch_size(self) -> int:
        """How many images to classify per scan step."""
        return 1

    @abstractmethod
    def glashelder_path(self, index: int) -> str | None:
        """Path for a glashelder index, or None if out of range."""

    @abstractmethod
    def artwork_path(self, index: int) -> str:
        """Reference for a artwork index, or "" if unknown/out of range.

        A local path or, for a scraped artwork, the image URL.
        """

    def artwork(self, index: int) -> dict | None:
        """The scraped record behind a reference, as JSON for the templates."""
        from ..batch import artwork_of

        found = artwork_of(self.artwork_path(index), getattr(self, "_artworks", None))
        return dataclasses.asdict(found) if found is not None else None

    @abstractmethod
    def classify(self, index: int, threshold: float) -> MatchItem | None:
        """One item, or None if index is out of range."""

    def classify_many(
        self, indices: list[int], threshold: float
    ) -> list[MatchItem | None]:
        return [self.classify(i, threshold) for i in indices]


def _current_keys(result: dict) -> dict:
    """A batch-classify result, also one written before `wikiart_*` became `artwork_*`."""
    if "artwork_index" in result:
        return result
    return {
        **result,
        "artwork_index": result["wikiart_index"],
        "artwork_path": result["wikiart_path"],
    }


class PrecomputedSource(ClassificationSource):
    """Serves classification items from pre-computed batch_classify results."""

    def __init__(
        self,
        results: list[dict],
        glashelder_root,
        reference_root,
        artworks: dict[str, Artwork] | None = None,
    ):
        self._results = [_current_keys(r) for r in results]
        results = self._results
        self._artworks = artworks or {}

        glashelder_paths = [r["glashelder_path"] for r in results]
        self._glashelder_paths = rebase_paths(glashelder_paths, glashelder_root)

        artwork_indices: dict[int, str] = {}
        for r in results:
            artwork_indices[r["artwork_index"]] = r["artwork_path"]
        keys = list(artwork_indices.keys())
        rebased_values = rebase_paths(
            [artwork_indices[k] for k in keys], reference_root
        )
        self._artwork_paths: dict[int, str] = dict(
            zip(keys, rebased_values, strict=True)
        )

    @property
    def total_images(self) -> int:
        return len(self._glashelder_paths)

    def glashelder_path(self, index: int) -> str | None:
        if index < 0 or index >= len(self._glashelder_paths):
            return None
        return self._glashelder_paths[index]

    def artwork_path(self, index: int) -> str:
        return self._artwork_paths.get(index, "")

    def classify(self, index: int, threshold: float) -> MatchItem | None:
        if index < 0 or index >= len(self._results):
            return None

        r = self._results[index]
        confidence = r["confidence"]
        is_match = confidence > threshold

        return MatchItem(
            glashelder_index=index,
            glashelder_path=r["glashelder_path"],
            top_indices=[r["artwork_index"]],
            similarities=r["similarities"],
            is_match=is_match,
            diff=confidence,
            threshold=threshold,
            classifier_type="precomputed",
            features=r.get("features", {}),
        )


class LiveSource(ClassificationSource):
    """Classifies glashelder images against reference embeddings on demand.

    The embedding search for every glashelder image is done once, up front, in a
    single pass over the references: they may not fit in RAM, and a pass per
    image would then read them all from disk per image. After that, only the
    small top-k arrays are kept, and each request is a lookup plus, with a
    classifier, the ORB/hash features of its best match.
    """

    def __init__(
        self,
        inputs: MatchingInputs,
        *,
        top_k: int = 10,
        classifier: MLClassifier | None = None,
        search_workers: int = 8,
        search_batch_size: int = 32,
        artworks: dict[str, Artwork] | None = None,
        candidates: int = 1,
    ):
        from ..loading import top_matches

        self._artworks = artworks or {}
        self._glashelder_paths = inputs.glashelder_paths
        self._artwork_paths = inputs.artwork_paths

        started = time.monotonic()
        self._similarities, self._indices = top_matches(inputs, top_k)
        logger.info(
            f"top {top_k} of {len(self._artwork_paths):,} references for "
            f"{len(self._glashelder_paths):,} images ready in "
            f"{time.monotonic() - started:.1f}s"
        )
        self._classifier = classifier
        self._search_workers = search_workers
        self._search_batch_size = search_batch_size
        self._candidates = max(1, candidates)
        self._cache: dict[int, MatchItem] = {}
        # the ML classifier's verdict per candidate tried, per image: kept across
        # threshold changes, since a higher threshold may need more candidates
        self._tried: dict[int, list[tuple[float, dict[str, float]]]] = {}
        self._pool: ProcessPoolExecutor | None = None

    @property
    def total_images(self) -> int:
        return len(self._glashelder_paths)

    @property
    def batch_size(self) -> int:
        return self._search_batch_size if self._classifier is not None else 1

    def _extract(self, work_items: list) -> dict[int, dict[str, float]]:
        """ORB/hash features in the server's worker pool, started on first use.

        One pool for the life of the server: a new one per batch would fork every
        worker again for each 32 images. A pool that broke (a worker died) is
        replaced once.
        """
        from concurrent.futures.process import BrokenProcessPool

        from .. import batch

        def run() -> dict[int, dict[str, float]]:
            if self._pool is None:
                self._pool = batch.pool(self._search_workers)
            return batch.extract_features_batch(
                work_items, self._search_workers, progress=False, executor=self._pool
            )

        try:
            return run()
        except BrokenProcessPool:
            logger.warning("the feature worker pool broke; starting a new one")
            self._pool = None
            return run()

    def glashelder_path(self, index: int) -> str | None:
        if index < 0 or index >= len(self._glashelder_paths):
            return None
        return self._glashelder_paths[index]

    def artwork_path(self, index: int) -> str:
        return (
            self._artwork_paths[index] if 0 <= index < len(self._artwork_paths) else ""
        )

    def _embedding_results(self, index: int) -> tuple[list[int], list[float]]:
        """The precomputed top-k references and similarities for an image."""
        return self._indices[index].tolist(), self._similarities[index].tolist()

    def classify(self, index: int, threshold: float) -> MatchItem | None:
        if index < 0 or index >= self.total_images:
            return None
        if self._classifier is not None:
            return self.classify_many([index], threshold)[0]

        if index in self._cache:
            cached = self._cache[index]
            is_match = cached.diff > threshold
            return dataclasses.replace(cached, is_match=is_match, threshold=threshold)

        from ..similarity_classifier import SimpleClassifier

        top_indices_list, similarities = self._embedding_results(index)
        result = SimpleClassifier(threshold=threshold).classify(similarities)
        item = MatchItem(
            glashelder_index=index,
            glashelder_path=self._glashelder_paths[index],
            top_indices=top_indices_list,
            similarities=similarities,
            is_match=result.is_match,
            diff=result.diff,
            threshold=threshold,
            classifier_type="simple",
            top_similarities=similarities,
        )
        self._cache[index] = item
        return item

    def classify_many(
        self, indices: list[int], threshold: float
    ) -> list[MatchItem | None]:
        """The ML verdict for each image, trying up to `candidates` of its top
        matches while the ones before them are no match (see `batch.score_round`).

        Candidates already scored are kept, so going back to an image, or a lower
        threshold, costs nothing; a higher one only scores the extra candidates.
        """
        if self._classifier is None:
            return [self.classify(i, threshold) for i in indices]

        from .. import batch

        images = [i for i in indices if 0 <= i < self.total_images]
        while batch.score_round(
            self._tried,
            images,
            threshold=threshold,
            candidates=self._candidates,
            glashelder_paths=self._glashelder_paths,
            artwork_paths=self._artwork_paths,
            top_indices=self._indices,
            top_similarities=self._similarities,
            classifier=self._classifier,
            extract=self._extract,
        ):
            pass

        return [
            self._ml_item(i, threshold) if 0 <= i < self.total_images else None
            for i in indices
        ]

    def _ml_item(self, index: int, threshold: float) -> MatchItem:
        from .. import batch

        tried = self._tried[index]
        rank = batch.choose(tried, threshold, self._candidates)
        confidence, features = tried[rank]
        top_indices, top_similarities = self._embedding_results(index)
        return MatchItem(
            glashelder_index=index,
            glashelder_path=self._glashelder_paths[index],
            top_indices=top_indices,
            similarities=batch.shifted(top_similarities, rank),
            is_match=confidence > threshold,
            diff=confidence,
            threshold=threshold,
            classifier_type="ml",
            features=features,
            rank=rank,
            top_similarities=top_similarities,
        )
