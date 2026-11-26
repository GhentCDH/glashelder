"""Match results on disk, keyed by pair, safe to extend: a `.matches` directory.

    meta.json            version, matcher and the settings that change a score
    scores/*.parquet     the scores of matched pairs, one file per finished batch
    candidates.parquet   the candidate pairs of the last run, with their
                         retrieval rank and similarity
    decisions.csv        one decision per plate of the last run (see `decide`)
    matches.jsonl        the plates that match, for `unstir serve`

A pair is matched once. A new run, for example after references were added,
only matches the candidate pairs that no score file has yet. Each batch is
written in one step, so a run that stops loses at most the batch in progress.

The score of a pair depends only on its two images and the settings, not on
the other references, so it stays valid when the reference set grows. Rank and
similarity do change with the reference set, which is why they are in
`candidates.parquet` and are written again on every run.

Refs are stored relative to the plate and reference image directories, so a
store made on one machine works on another.
"""

import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.errors import UserError
from common.fs import atomic_write, atomic_writer
from common.images import is_url

from .labels import Key, resolve
from .match import Settings

META, SCORES, CANDIDATES = "meta.json", "scores", "candidates.parquet"
VERSION = 1
MATCHER = "aliked+lightglue"

logger = logging.getLogger(__name__)


def is_matches(path: str | Path) -> bool:
    return (Path(path) / META).is_file()


def relative(ref: str, root: Path) -> str:
    if is_url(ref):
        return ref
    try:
        return Path(ref).relative_to(root).as_posix()
    except ValueError:
        return ref


class MatchStore:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta

    @classmethod
    def open(cls, path: str | Path) -> "MatchStore":
        path = Path(path)
        if not is_matches(path):
            raise UserError(f"{path} is not a match store, run `unstir match` first")
        meta = json.loads((path / META).read_text())
        if meta.get("version") != VERSION:
            raise UserError(f"{path}: unsupported version {meta.get('version')}")
        return cls(path, meta)

    @classmethod
    def create_or_open(
        cls, path: str | Path, settings: Settings, *, rescore: bool
    ) -> "MatchStore":
        """The store at `path`, made when it is not there. Settings that differ
        from the stored ones make every score invalid: that needs `rescore`."""
        path = Path(path)
        scoring = {"matcher": MATCHER, "settings": asdict(settings)}
        if is_matches(path):
            store = cls.open(path)
            stored = {k: store.meta.get(k) for k in scoring}
            if stored != scoring:
                if not rescore:
                    raise UserError(
                        f"{path} was matched with {stored}, not {scoring}. Give "
                        "--rescore to throw its scores away and match again, or "
                        "use another --store"
                    )
                logger.warning(f"throwing away the scores in {path}")
                shutil.rmtree(path / SCORES, ignore_errors=True)
            meta = {**store.meta, **scoring}
        elif path.exists() and any(path.iterdir()):
            raise UserError(f"{path} exists and is not a match store")
        else:
            meta = {
                "version": VERSION,
                **scoring,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        (path / SCORES).mkdir(parents=True, exist_ok=True)
        atomic_write(path / META, json.dumps(meta, indent=2) + "\n")
        return cls(path, meta)

    def _parts(self) -> list[Path]:
        return sorted((self.path / SCORES).glob("*.parquet"))

    def _table(self, columns: list[str] | None = None) -> pa.Table | None:
        parts = self._parts()
        if not parts:
            return None
        return pa.concat_tables(pq.read_table(p, columns=columns) for p in parts)

    def scored(self) -> set[Key]:
        """The pairs that have a score, as stored (relative refs)."""
        table = self._table(["plate", "reference"])
        if table is None:
            return set()
        return set(
            zip(
                table["plate"].to_pylist(),
                table["reference"].to_pylist(),
                strict=True,
            )
        )

    def add(self, rows: list[dict]) -> None:
        """Commit the scores of one batch."""
        if not rows:
            return
        part = self.path / SCORES / f"{time.time_ns()}.parquet"
        with atomic_writer(part, binary=True) as file:
            pq.write_table(pa.Table.from_pylist(rows), file)

    def set_candidates(self, rows: list[dict]) -> None:
        with atomic_writer(self.path / CANDIDATES, binary=True) as file:
            pq.write_table(pa.Table.from_pylist(rows), file)

    def update_meta(self, **values) -> None:
        self.meta.update(values)
        atomic_write(self.path / META, json.dumps(self.meta, indent=2) + "\n")

    def rows(self, plate_root: Path, reference_root: Path) -> list[dict]:
        """The candidates of the last run with their scores, refs made readable
        on this machine. Candidates without a score are left out."""
        if not (self.path / CANDIDATES).is_file():
            raise UserError(f"{self.path} has no candidates, run `unstir match`")
        table = self._table()
        scores = {
            (row["plate"], row["reference"]): row
            for row in (table.to_pylist() if table is not None else [])
        }
        return [
            {
                **candidate,
                **scores[key],
                "plate": resolve(candidate["plate"], plate_root),
                "reference": resolve(candidate["reference"], reference_root),
            }
            for candidate in pq.read_table(self.path / CANDIDATES).to_pylist()
            if (key := (candidate["plate"], candidate["reference"])) in scores
        ]
