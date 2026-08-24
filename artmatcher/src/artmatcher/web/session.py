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

    def to_json(self) -> str:
        """The downloadable dataset JSON."""
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclass
class LabelSession:
    """Mutable state for one labeling session."""

    threshold: float
    order: list[int]  # traversal order (was `shuffled_indices`)
    queue_position: int = 0
    current_item: object | None = None  # will become MatchItem | None in a later step
    dataset: LabelDataset = field(default_factory=LabelDataset)
