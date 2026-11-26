"""The track record of which model assessed which image, kept across runs.

An append-only JSON Lines file with one line per attempt of a model on an
image. Every line is flushed to disk as soon as it is written, so a job killed
at its time limit loses at most the image it was busy with, and the next run
picks up where it left off: an image a model assessed is never sent to that
model again.

Images are identified by their path relative to the folder that was assessed,
so one ledger belongs to one image folder.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from glaswacht.models.safety_model import Rating

logger = logging.getLogger(__name__)

# attempts that ended in an exception (out of memory, an unreadable file, ...)
# after which a model gives up on an image
MAX_ATTEMPTS = 3

DONE = "done"
FAILED = "failed"

# environment variables HPC schedulers name the current job in
_JOB_VARIABLES = ("SLURM_JOB_ID", "PBS_JOBID")


def _job_id() -> str:
    return next((os.environ[v] for v in _JOB_VARIABLES if v in os.environ), "")


@dataclass
class _Entry:
    """What the ledger knows about one model on one image."""

    rating: Rating | None = None
    failures: int = 0
    errors: list[str] = field(default_factory=list)


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._read()
        self._end_last_line()

    def _end_last_line(self) -> None:
        """End a line cut off mid-write, so the next one starts on its own
        line instead of being glued onto the broken one."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    def _read(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (record["image"], record["model"])
                    status = record["status"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # a line cut off when a job was killed mid-write
                    logger.warning(
                        "%s:%d: skipping an unreadable ledger line", self.path, number
                    )
                    continue
                entry = self._entries.setdefault(key, _Entry())
                if status == DONE:
                    entry.rating = Rating(
                        rating=record.get("rating", "Unknown"),
                        category=record.get("category", ""),
                        rationale=record.get("rationale", ""),
                        image=record["image"],
                    )
                elif status == FAILED:
                    entry.failures += 1
                    entry.errors.append(record.get("error", ""))

    def rating(self, image: str, model: str) -> Rating | None:
        """The assessment `model` made of `image`, or None when it has none."""
        entry = self._entries.get((image, model))
        return entry.rating if entry else None

    def failures(self, image: str, model: str) -> int:
        entry = self._entries.get((image, model))
        return entry.failures if entry else 0

    def gave_up(self, image: str, model: str) -> bool:
        """Whether `model` failed on `image` too often to try again."""
        return (
            self.rating(image, model) is None
            and self.failures(image, model) >= MAX_ATTEMPTS
        )

    def pending(self, image: str, models: list[str]) -> list[str]:
        """The `models` that still have to assess `image`: neither assessed
        it nor gave up on it."""
        return [
            m
            for m in models
            if self.rating(image, m) is None and not self.gave_up(image, m)
        ]

    def images(self) -> list[str]:
        """Every image the ledger has a line for, in the order first seen."""
        return list(dict.fromkeys(image for image, _ in self._entries))

    def models(self) -> list[str]:
        """Every model the ledger has a line for, in the order first seen."""
        return list(dict.fromkeys(model for _, model in self._entries))

    def record_done(self, model: str, rating: Rating, **context: object) -> None:
        """Record that `model` assessed `rating.image`. An answer of "Error"
        or "Unknown" is still an assessment: the model ran, so it is not
        asked again."""
        self._entries.setdefault((rating.image, model), _Entry()).rating = rating
        self._append(
            image=rating.image,
            model=model,
            status=DONE,
            rating=rating.rating,
            category=rating.category,
            rationale=rating.rationale,
            **context,
        )

    def record_failure(
        self, image: str, model: str, error: str, **context: object
    ) -> None:
        """Record that `model` raised an exception on `image`; it is tried
        again in a later run, up to MAX_ATTEMPTS times."""
        entry = self._entries.setdefault((image, model), _Entry())
        entry.failures += 1
        entry.errors.append(error)
        self._append(image=image, model=model, status=FAILED, error=error, **context)

    def _append(self, **record) -> None:
        record = {
            **record,
            "time": datetime.now().isoformat(timespec="seconds"),
            "job": _job_id(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
