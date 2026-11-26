"""Pairs to label, and the labels given to them.

A pairs file is JSONL, one pair per line:

    {"plate": "<ref>", "reference": "<ref>", ...any other fields, such as scores}

A ref is an image path or URL. A relative path is relative to the pairs file.

The labels file is JSONL too, and only grows: one line per label given,

    {"plate": "<ref>", "reference": "<ref>", "label": "match", "time": "..."}

with the refs as they are in the pairs file. The last line for a pair wins, and a
`null` label clears it. So a crash loses at most the line being written, and the
file keeps every change.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from common.errors import UserError
from common.images import is_url

Label = Literal["match", "no_match"]
LABELS: tuple[Label, ...] = ("match", "no_match")

Key = tuple[str, str]


@dataclass(frozen=True)
class Pair:
    plate: str
    reference: str
    info: dict = field(default_factory=dict)
    """The other fields of the line in the pairs file."""

    @property
    def key(self) -> Key:
        return self.plate, self.reference


def resolve(ref: str, base: Path) -> str:
    """`ref` as a path or URL this machine can read."""
    if is_url(ref) or Path(ref).is_absolute():
        return ref
    return str(base / ref)


def read_pairs(path: Path) -> list[Pair]:
    """The pairs in `path`, in file order, without duplicates."""
    pairs: dict[Key, Pair] = {}
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                plate, reference = row.pop("plate"), row.pop("reference")
            except KeyError as error:
                raise UserError(
                    f"{path}:{number}: a pair needs `plate` and `reference`"
                ) from error
            pairs.setdefault((plate, reference), Pair(plate, reference, row))
    return list(pairs.values())


class Labels:
    """The labels in a labels file, which `give` appends to."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.labels: dict[Key, Label] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                for line in filter(str.strip, file):
                    self.apply(json.loads(line))

    def apply(self, row: dict) -> None:
        key = row["plate"], row["reference"]
        if row["label"] is None:
            self.labels.pop(key, None)
        else:
            self.labels[key] = row["label"]

    def give(self, pair: Pair, label: Label | None) -> None:
        row = {
            "plate": pair.plate,
            "reference": pair.reference,
            "label": label,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            self.apply(row)

    def get(self, pair: Pair) -> Label | None:
        return self.labels.get(pair.key)
