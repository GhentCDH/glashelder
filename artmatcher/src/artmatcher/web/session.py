import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass, field

_BUCKETS = {
    "positives": (True, True),
    "negatives": (True, False),
    "false_positives": (False, True),
    "false_negatives": (False, False),
}
"""Each bucket as (the classifier was right, the classifier called a match)."""

BUCKETS = tuple(_BUCKETS)


def was_match(bucket: str) -> bool:
    """Whether the classifier had called the pairs in a bucket a match."""
    return _BUCKETS[bucket][1]


def truth(bucket: str) -> bool:
    """Whether the pairs in a bucket truly match, by their label."""
    correct, said_match = _BUCKETS[bucket]
    return correct == said_match


def bucket_for(correct: bool, was_match: bool) -> str:
    """The bucket of a label: was the classifier right, and did it call a match."""
    return next(name for name, key in _BUCKETS.items() if key == (correct, was_match))


@dataclass
class LabelDataset:
    """The four buckets of labeled examples collected during a session."""

    positives: list[dict] = field(default_factory=list)
    negatives: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    false_negatives: list[dict] = field(default_factory=list)

    def bucket(self, name: str) -> list[dict]:
        return getattr(self, name)

    def counts(self) -> dict[str, int]:
        """Per-bucket counts plus a "total" key."""
        counts = {name: len(self.bucket(name)) for name in BUCKETS}
        return counts | {"total": sum(counts.values())}

    def entries(self) -> list[dict]:
        """Every entry across all four buckets."""
        return [entry for name in BUCKETS for entry in self.bucket(name)]

    def labeled(self) -> Iterator[tuple[dict, bool]]:
        """Every entry with whether its pair truly matches."""
        return ((entry, truth(name)) for name in BUCKETS for entry in self.bucket(name))

    def forget(self, glashelder_index: int) -> None:
        """Drop every label of an image."""
        for name in BUCKETS:
            self.bucket(name)[:] = [
                entry
                for entry in self.bucket(name)
                if entry.get("glashelder_index") != glashelder_index
            ]

    def as_dict(self) -> dict[str, list[dict]]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """The downloadable dataset JSON."""
        return json.dumps(self.as_dict(), indent=2)


@dataclass
class LabelSession:
    """Mutable state for one labeling session.

    `labeled` holds the glashelder indices that already have a label. The
    traversal skips them, which is what makes every image in the sample
    reachable instead of only the ones the cursor happened to land on.

    `order` may cover only part of the glashelder images (`serve --subset`). The
    indices stay those of the full set, so labels from different subsets agree;
    `labeled` can then hold images outside `order`, which the counts leave out.
    """

    session_id: str
    threshold: float
    order: list[int]  # traversal order
    queue_position: int = 0
    current_item: object | None = None
    dataset: LabelDataset = field(default_factory=LabelDataset)
    labeled: set[int] = field(default_factory=set)
    created_at: str | None = None

    # Which kind of image the labeler is walking through: set by the last
    # Next Match / Next Negative press, and reused after every label so the
    # labeler keeps serving the kind that is actually being worked on.
    want_match: bool = True

    # Bumped by every navigating request. A scan that discovers a newer request
    # has started is superseded: it stops early and keeps its result to itself,
    # so a slow search cannot overwrite the image a later click already put on
    # screen (which would otherwise attach the next label to the wrong image).
    request_seq: int = 0

    # Where a running search for the next image has got to in `order`, so the
    # page can show it move while the search is still going. None when idle.
    # Not saved: it only means something while a request is in flight.
    scan_position: int | None = None

    def begin_request(self) -> int:
        self.request_seq += 1
        return self.request_seq

    def is_current(self, seq: int | None) -> bool:
        return seq is None or seq == self.request_seq

    # Threshold at which a scan for a match / a negative last came up empty.
    # Labeling only ever removes candidates, so the answer stays valid until the
    # threshold changes -- which keeps the endgame from rescanning (and, in live
    # mode, re-classifying) the whole queue on every single label.
    exhausted_matches_at: float | None = None
    exhausted_negatives_at: float | None = None

    def note_exhausted(self, want_match: bool) -> None:
        if want_match:
            self.exhausted_matches_at = self.threshold
        else:
            self.exhausted_negatives_at = self.threshold

    def known_exhausted(self, want_match: bool) -> bool:
        seen_at = (
            self.exhausted_matches_at if want_match else self.exhausted_negatives_at
        )
        return seen_at is not None and seen_at == self.threshold

    def set_threshold(self, threshold: float) -> None:
        """Change the threshold, invalidating the exhaustion memos."""
        self.threshold = threshold
        self.exhausted_matches_at = None
        self.exhausted_negatives_at = None

    def __post_init__(self) -> None:
        self._members = set(self.order)

    def includes(self, index: int) -> bool:
        """Whether an image is part of this session's order."""
        return index in self._members

    @property
    def total_images(self) -> int:
        return len(self.order)

    @property
    def labeled_count(self) -> int:
        """Labeled images in this session's order (others may carry labels too)."""
        return len(self.labeled & self._members)

    @property
    def remaining(self) -> int:
        return len(self.order) - self.labeled_count

    def rebuild_labeled(self) -> None:
        """Recompute `labeled` from the dataset entries (used after loading)."""
        self.labeled = {
            entry["glashelder_index"]
            for entry in self.dataset.entries()
            if "glashelder_index" in entry
        }
