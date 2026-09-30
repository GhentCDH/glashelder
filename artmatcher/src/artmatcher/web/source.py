from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from embed import rebase_paths

if TYPE_CHECKING:
    from artscraper import Artwork

    from ..loading import MatchingInputs
    from ..ml_classifier import MLClassifier


@dataclass
class _EmbeddingData:
    batch_idx: int
    image_index: int
    top_indices_list: list[int]
    similarities: list[float]
    glashelder_path: str
    wikiart_path: str


@dataclass
class MatchItem:
    glashelder_index: int
    glashelder_path: str
    top_indices: list[int]
    similarities: list[float]
    is_match: bool
    diff: float
    threshold: float
    classifier_type: str
    features: dict[str, float] = field(default_factory=dict)


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
    def wikiart_path(self, index: int) -> str:
        """Reference for a wikiart index, or "" if unknown/out of range.

        A local path or, for a scraped artwork, the image URL.
        """

    def artwork(self, index: int) -> dict | None:
        """The scraped record behind a reference, as JSON for the templates."""
        from ..batch import artwork_of

        found = artwork_of(self.wikiart_path(index), getattr(self, "_artworks", None))
        return dataclasses.asdict(found) if found is not None else None

    @abstractmethod
    def classify(self, index: int, threshold: float) -> MatchItem | None:
        """One item, or None if index is out of range."""

    def classify_many(
        self, indices: list[int], threshold: float
    ) -> list[MatchItem | None]:
        return [self.classify(i, threshold) for i in indices]


class PrecomputedSource(ClassificationSource):
    """Serves classification items from pre-computed batch_classify results."""

    def __init__(
        self,
        results: list[dict],
        glashelder_root,
        reference_root,
        artworks: dict[str, Artwork] | None = None,
    ):
        self._results = results
        self._artworks = artworks or {}

        glashelder_paths = [r["glashelder_path"] for r in results]
        self._glashelder_paths = rebase_paths(glashelder_paths, glashelder_root)

        wikiart_indices: dict[int, str] = {}
        for r in results:
            wikiart_indices[r["wikiart_index"]] = r["wikiart_path"]
        keys = list(wikiart_indices.keys())
        rebased_values = rebase_paths(
            [wikiart_indices[k] for k in keys], reference_root
        )
        self._wikiart_paths: dict[int, str] = dict(
            zip(keys, rebased_values, strict=True)
        )

    @property
    def total_images(self) -> int:
        return len(self._glashelder_paths)

    def glashelder_path(self, index: int) -> str | None:
        if index < 0 or index >= len(self._glashelder_paths):
            return None
        return self._glashelder_paths[index]

    def wikiart_path(self, index: int) -> str:
        return self._wikiart_paths.get(index, "")

    def classify(self, index: int, threshold: float) -> MatchItem | None:
        if index < 0 or index >= len(self._results):
            return None

        r = self._results[index]
        confidence = r["confidence"]
        is_match = confidence > threshold

        return MatchItem(
            glashelder_index=index,
            glashelder_path=r["glashelder_path"],
            top_indices=[r["wikiart_index"]],
            similarities=r["similarities"],
            is_match=is_match,
            diff=confidence,
            threshold=threshold,
            classifier_type="precomputed",
            features=r.get("features", {}),
        )


class LiveSource(ClassificationSource):
    """Classifies glashelder images against reference embeddings on demand.

    Note: this class holds GPU tensors (embeddings) as instance attributes
    while handing work off to a ProcessPoolExecutor for ORB/hash feature
    extraction. This is fine under the "fork" multiprocessing start method
    (the default on Linux) but would break under "spawn", since the CUDA
    tensors would need to be re-pickled into the worker processes.
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
    ):
        self._artworks = artworks or {}
        self._glashelder_emb = inputs.glashelder_emb
        self._glashelder_paths = inputs.glashelder_paths
        self._wikiart_emb = inputs.wikiart_emb
        self._wikiart_paths = inputs.wikiart_paths
        self._top_k = top_k
        self._classifier = classifier
        self._search_workers = search_workers
        self._search_batch_size = search_batch_size
        self._cache: dict[int, MatchItem] = {}

    @property
    def total_images(self) -> int:
        return len(self._glashelder_paths)

    @property
    def batch_size(self) -> int:
        return self._search_batch_size if self._classifier is not None else 1

    def glashelder_path(self, index: int) -> str | None:
        if index < 0 or index >= len(self._glashelder_paths):
            return None
        return self._glashelder_paths[index]

    def wikiart_path(self, index: int) -> str:
        return (
            self._wikiart_paths[index] if 0 <= index < len(self._wikiart_paths) else ""
        )

    def _embedding_results(self, index: int) -> tuple[list[int], list[float]]:
        """Get embedding search results for an image (fast)."""
        import cluster

        emb = self._glashelder_emb[index]
        top_indices, top_values = cluster.perform_search(
            self._top_k, emb.unsqueeze(0), self._wikiart_emb, self._wikiart_paths
        )
        return top_indices.cpu().numpy().tolist(), top_values.cpu().numpy().tolist()

    def classify(self, index: int, threshold: float) -> MatchItem | None:
        if index < 0 or index >= self.total_images:
            return None

        if index in self._cache:
            cached = self._cache[index]
            is_match = cached.diff > threshold
            return dataclasses.replace(cached, is_match=is_match, threshold=threshold)

        top_indices_list, similarities = self._embedding_results(index)

        glashelder_path = self._glashelder_paths[index]
        wikiart_path = (
            self._wikiart_paths[top_indices_list[0]] if top_indices_list else None
        )

        if self._classifier is not None:
            self._classifier.threshold = threshold
            result = self._classifier.classify(
                similarities, glashelder_path, wikiart_path
            )
            item = MatchItem(
                glashelder_index=index,
                glashelder_path=glashelder_path,
                top_indices=top_indices_list,
                similarities=similarities,
                is_match=result.is_match,
                diff=result.confidence,
                threshold=threshold,
                classifier_type="ml",
                features=result.features,
            )
        else:
            from ..similarity_classifier import SimpleClassifier

            simple_clf = SimpleClassifier(threshold=threshold)
            result = simple_clf.classify(similarities)
            item = MatchItem(
                glashelder_index=index,
                glashelder_path=glashelder_path,
                top_indices=top_indices_list,
                similarities=similarities,
                is_match=result.is_match,
                diff=result.diff,
                threshold=threshold,
                classifier_type="simple",
            )

        self._cache[index] = item
        return item

    def classify_many(
        self, indices: list[int], threshold: float
    ) -> list[MatchItem | None]:
        if self._classifier is None:
            return [self.classify(i, threshold) for i in indices]

        from .. import batch

        results: list[MatchItem | None] = [None] * len(indices)

        uncached_batch = []
        for batch_idx, image_index in enumerate(indices):
            if image_index in self._cache:
                cached = self._cache[image_index]
                is_match = cached.diff > threshold
                results[batch_idx] = dataclasses.replace(
                    cached, is_match=is_match, threshold=threshold
                )
            else:
                uncached_batch.append((batch_idx, image_index))

        if not uncached_batch:
            return results

        embedding_data = []
        for batch_idx, image_index in uncached_batch:
            if image_index < 0 or image_index >= self.total_images:
                embedding_data.append(None)
                continue
            top_indices_list, similarities = self._embedding_results(image_index)
            glashelder_path = self._glashelder_paths[image_index]
            wikiart_path = (
                self._wikiart_paths[top_indices_list[0]] if top_indices_list else ""
            )
            embedding_data.append(
                _EmbeddingData(
                    batch_idx=batch_idx,
                    image_index=image_index,
                    top_indices_list=top_indices_list,
                    similarities=similarities,
                    glashelder_path=glashelder_path,
                    wikiart_path=wikiart_path,
                )
            )

        work_items = [
            (i, d.glashelder_path, d.wikiart_path, d.similarities, 0)
            for i, d in enumerate(embedding_data)
            if d is not None
        ]

        features_map = batch.extract_features_batch(
            work_items, self._search_workers, progress=False
        )

        confidences = batch.predict_confidences(
            self._classifier,
            features_map,
            list(range(len(work_items))),
        )

        confidence_by_pos = dict(zip(range(len(work_items)), confidences, strict=True))

        pos = 0
        for i, data in enumerate(embedding_data):
            if data is None:
                continue
            confidence = confidence_by_pos[pos]
            pos += 1

            is_match = confidence > threshold

            item = MatchItem(
                glashelder_index=data.image_index,
                glashelder_path=data.glashelder_path,
                top_indices=data.top_indices_list,
                similarities=data.similarities,
                is_match=is_match,
                diff=confidence,
                threshold=threshold,
                classifier_type="ml",
                features=features_map[i],
            )
            results[data.batch_idx] = item
            self._cache[data.image_index] = item

        return results
