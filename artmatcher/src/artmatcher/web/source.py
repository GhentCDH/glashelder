from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from embed import rebase_paths


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
        """Path for a wikiart index, or "" if unknown/out of range."""

    @abstractmethod
    def classify(self, index: int, threshold: float) -> MatchItem | None:
        """One item, or None if index is out of range."""

    def classify_many(
        self, indices: list[int], threshold: float
    ) -> list[MatchItem | None]:
        return [self.classify(i, threshold) for i in indices]


class PrecomputedSource(ClassificationSource):
    """Serves classification items from pre-computed batch_classify results."""

    def __init__(self, results: list[dict], glashelder_root, reference_root):
        self._results = results

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
        is_match = confidence >= threshold

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
