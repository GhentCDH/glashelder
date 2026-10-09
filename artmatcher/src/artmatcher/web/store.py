"""On-disk persistence for labeling sessions.

One JSON file per session, rewritten after every label so a crash, a restart,
or a closed browser never loses collected work.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from common.fs import atomic_writer

from .session import BUCKETS, LabelDataset, LabelSession, was_match

logger = logging.getLogger(__name__)

# Session ids reach us from the client (localStorage / query string) and are
# used as filenames, so only this shape is ever accepted.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_RE.match(session_id))


def slugify_session_name(name: str) -> str:
    """Turn a name typed by a person into a usable session id.

    Session names double as filenames, so spaces become dashes and anything
    outside the allowed set is dropped. Returns "" when nothing usable is left,
    which callers report back rather than silently inventing a name.
    """
    slug = re.sub(r"\s+", "-", name.strip())
    slug = re.sub(r"[^A-Za-z0-9_-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug[:64]


def artwork_keys(record: dict) -> dict:
    """A copy of a label or a batch-classify result with the `wikiart_*` keys
    of files written before they became `artwork_*` renamed."""

    def current(key: str) -> str:
        if key.startswith("wikiart_"):
            return "artwork_" + key.removeprefix("wikiart_")
        return key

    return {
        current(key): value
        for key, value in record.items()
        if current(key) == key or current(key) not in record
    }


def normalize_dataset(data: dict) -> LabelDataset:
    """Read a dataset out of either export format.

    Accepts a session file written by this app (four buckets nested under
    "dataset") and the older plain "Download dataset" export (the four buckets
    at the top level). Entries are left as they are apart from filling in
    `is_match`, which older exports imply through the bucket rather than store,
    and renaming the `wikiart_*` keys of older labels to `artwork_*`.
    """
    buckets = data.get("dataset") if isinstance(data.get("dataset"), dict) else data
    if not isinstance(buckets, dict):
        raise ValueError("Not a labeling dataset: expected a JSON object")

    if not any(key in buckets for key in BUCKETS):
        raise ValueError(
            f"Not a labeling dataset: expected at least one of {', '.join(BUCKETS)}"
        )

    dataset = LabelDataset()
    for name in BUCKETS:
        entries = buckets.get(name) or []
        if not isinstance(entries, list):
            raise ValueError(f"Bucket {name!r} is not a list")
        normalized = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Bucket {name!r} contains a non-object entry")
            entry = artwork_keys(entry)
            # older exports imply `is_match` through the bucket
            entry.setdefault("is_match", was_match(name))
            normalized.append(entry)
        setattr(dataset, name, normalized)

    return dataset


def dataset_threshold(dataset: LabelDataset, default: float) -> float:
    """The threshold the imported entries were labeled at, if they record one."""
    thresholds = [
        entry["threshold"]
        for entry in dataset.entries()
        if isinstance(entry.get("threshold"), int | float)
    ]
    return float(thresholds[-1]) if thresholds else default


class SessionStore:
    """Reads and writes labeling sessions as JSON files in one directory."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        if not is_valid_session_id(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.directory / f"{session_id}.json"

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).is_file()

    def load(self, session_id: str, order: list[int], threshold: float):
        """Load a session, or None when it has never been saved.

        `order` and `threshold` are the defaults for a session that exists but
        predates a field; the stored threshold wins when present.
        """
        path = self.path_for(session_id)
        if not path.is_file():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read session {session_id} from {path}: {e}")
            return None

        # Goes through the same normalizer as an upload, so a file copied into
        # this directory by hand works whichever export format it came from.
        try:
            dataset = normalize_dataset(data)
        except ValueError as e:
            logger.error(f"Session file {path} is not a labeling dataset: {e}")
            return None

        session = LabelSession(
            session_id=session_id,
            threshold=float(data.get("threshold", threshold)),
            order=order,
            dataset=dataset,
            created_at=data.get("created_at"),
            want_match=bool(data.get("want_match", True)),
        )
        session.rebuild_labeled()

        logger.info(
            f"Resumed session {session_id}: {session.labeled_count} already labeled, "
            f"{session.remaining} to go"
        )
        return session

    def save(self, session: LabelSession) -> None:
        """Write the session atomically, so an interrupted write cannot corrupt it."""
        path = self.path_for(session.session_id)
        now = datetime.now().isoformat(timespec="seconds")
        if session.created_at is None:
            session.created_at = now

        payload = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": now,
            "threshold": session.threshold,
            "want_match": session.want_match,
            "total_images": session.total_images,
            "labeled_count": session.labeled_count,
            "dataset": session.dataset.as_dict(),
        }

        try:
            with atomic_writer(path) as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            logger.error(f"Could not save session {session.session_id} to {path}: {e}")

    def rename(self, old_id: str, new_id: str) -> None:
        """Move a session's file to a new id.

        Raises ValueError when either id is unusable or the target already
        exists -- renaming must never quietly overwrite someone's labels.
        """
        old_path = self.path_for(old_id)
        new_path = self.path_for(new_id)
        if new_path.exists():
            raise ValueError(f"A session named {new_id!r} already exists")

        if old_path.is_file():
            os.replace(old_path, new_path)
            # The id is stored inside the file too, so keep the two in step.
            try:
                with open(new_path) as f:
                    data = json.load(f)
                data["session_id"] = new_id
                with atomic_writer(new_path) as f:
                    json.dump(data, f, indent=2)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(
                    f"Renamed {old_id} to {new_id} but could not rewrite it: {e}"
                )

        logger.info(f"Renamed session {old_id} to {new_id}")

    def list_sessions(self) -> list[dict]:
        """Summaries of every stored session, most recently updated first."""
        summaries = []
        for path in self.directory.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning(f"Skipping unreadable session file {path}")
                continue
            summaries.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "updated_at": data.get("updated_at", ""),
                    "labeled_count": data.get("labeled_count", 0),
                    "total_images": data.get("total_images", 0),
                }
            )
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries
