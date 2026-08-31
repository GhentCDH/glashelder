"""Content-hash based duplicate detection against images already imported
into a Label Studio project.

FILENAME_FIELD (see importer.py) makes future uploads matchable by name, but
tasks imported before that field existed — e.g. project 75's original 1370
tasks — carry only a bare base64 data: URI with no filename anywhere in the
task. The only way to recognize those is to decode the embedded image and
compare its content hash against a local file's hash.

Fetching every task is the expensive part (each one carries its full
embedded image, and Label Studio has no lighter "list tasks" endpoint), so
the result is cached to a JSON file (DedupeCache) that's built once via the
`known-images` CLI command and re-read locally (no network) by `import` and
`pipeline` afterwards.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import click

from labelstudio_metadata_import.importer import FILENAME_FIELD, LabelStudioImporter

_SAVE_EVERY = 200


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_of_data_uri(data_uri: str) -> str | None:
    """SHA-256 of the decoded image bytes in a data: URI, or None if DATA_URI
    isn't one (e.g. a plain uploaded-file path from the `sample` flow).
    """
    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        return None
    _, _, b64 = data_uri.partition(",")
    if not b64:
        return None
    return hashlib.sha256(base64.b64decode(b64)).hexdigest()


@dataclass
class DedupeCache:
    path: Path
    project_id: int | None = None
    seen_task_ids: set[int] = field(default_factory=set)
    hashes: dict[str, int] = field(default_factory=dict)  # sha256 -> task_id
    filenames: dict[str, int] = field(default_factory=dict)  # filename -> task_id

    @classmethod
    def load(cls, path: str | Path) -> DedupeCache:
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            project_id=data.get("project_id"),
            seen_task_ids=set(data.get("seen_task_ids", [])),
            hashes=data.get("hashes", {}),
            filenames=data.get("filenames", {}),
        )

    def save(self) -> None:
        data = {
            "project_id": self.project_id,
            "seen_task_ids": sorted(self.seen_task_ids),
            "hashes": self.hashes,
            "filenames": self.filenames,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def contains_file(self, local_path: Path) -> bool:
        return sha256_of_file(local_path) in self.hashes

    def contains_filename(self, filename: str) -> bool:
        return filename in self.filenames

    def refresh(self, importer: LabelStudioImporter, page_size: int | None = None) -> int:
        """Fetch every task in IMPORTER's project not already in this cache,
        hash its embedded image, and record it. Safe to interrupt and
        re-run: already-seen task ids are skipped, and progress is saved
        every _SAVE_EVERY newly-hashed tasks.

        Returns the number of newly-hashed tasks this call added.
        """
        if self.project_id is not None and self.project_id != importer.project_id:
            raise ValueError(
                f"Cache at {self.path} belongs to project {self.project_id}, "
                f"not {importer.project_id}. Use a different cache path."
            )
        self.project_id = importer.project_id

        total = importer.task_count()
        kwargs = {} if page_size is None else {"page_size": page_size}
        new_count = 0
        with click.progressbar(
            importer.iter_tasks(**kwargs),
            length=total,
            label="Fetching + hashing tasks",
            show_pos=True,
        ) as bar:
            for task in bar:
                task_id = task["id"]
                if task_id in self.seen_task_ids:
                    continue

                data = task.get("data", {})
                digest = sha256_of_data_uri(data.get("image", ""))
                if digest:
                    self.hashes[digest] = task_id
                filename = data.get(FILENAME_FIELD)
                if filename:
                    self.filenames[filename] = task_id

                self.seen_task_ids.add(task_id)
                new_count += 1
                if new_count % _SAVE_EVERY == 0:
                    self.save()

        self.save()
        return new_count
