import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class LabelDataset:
    """The four buckets of labeled examples collected during a session."""

    positives: list[dict] = field(default_factory=list)
    negatives: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)
    false_negatives: list[dict] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Per-bucket counts plus a "total" key."""
        return {
            "positives": len(self.positives),
            "negatives": len(self.negatives),
            "false_positives": len(self.false_positives),
            "false_negatives": len(self.false_negatives),
            "total": len(self.positives)
            + len(self.negatives)
            + len(self.false_positives)
            + len(self.false_negatives),
        }

    def entries(self) -> list[dict]:
        """Every entry across all four buckets."""
        return [
            *self.positives,
            *self.negatives,
            *self.false_positives,
            *self.false_negatives,
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
    """

    session_id: str
    threshold: float
    order: list[int]  # traversal order
    queue_position: int = 0
    current_item: object | None = None
    dataset: LabelDataset = field(default_factory=LabelDataset)
    labeled: set[int] = field(default_factory=set)
    created_at: str | None = None

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

    @property
    def total_images(self) -> int:
        return len(self.order)

    @property
    def labeled_count(self) -> int:
        return len(self.labeled)

    @property
    def remaining(self) -> int:
        return len(self.order) - len(self.labeled)

    def rebuild_labeled(self) -> None:
        """Recompute `labeled` from the dataset entries (used after loading)."""
        self.labeled = {
            entry["glashelder_index"]
            for entry in self.dataset.entries()
            if "glashelder_index" in entry
        }
