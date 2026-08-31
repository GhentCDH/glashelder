"""Resumable checkpoint state for long Glaswacht runs: tracks which images
have already been assessed (so a crash or restart doesn't redo GPU work) and
keeps a running record of every row produced, Safe or flagged alike.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
RESULTS_FILENAME = "results.csv"
NEEDS_RESEARCH_FILENAME = "needs_research.csv"


@dataclass
class Checkpoint:
    checkpoint_dir: Path
    model_names: list[str]
    done_images: set[str] = field(default_factory=set)

    @property
    def manifest_path(self) -> Path:
        return self.checkpoint_dir / MANIFEST_FILENAME

    @property
    def results_csv_path(self) -> Path:
        return self.checkpoint_dir / RESULTS_FILENAME

    @property
    def needs_research_csv_path(self) -> Path:
        return self.checkpoint_dir / NEEDS_RESEARCH_FILENAME

    def mark_chunk_done(
        self,
        image_names: list[str],
        rows: list[dict],
        needs_research_rows: list[dict] | None = None,
    ) -> None:
        """Persist that IMAGE_NAMES are fully assessed: append ROWS (every
        row this chunk produced — flagged, safe, or needing research) to the
        running results CSV, and separately append NEEDS_RESEARCH_ROWS (rows
        where a model returned something other than Safe/Potentially
        Problematic/Unsafe — a parse failure, not a real judgment) to their
        own CSV, since those are never sent to Label Studio and need a human
        to look at them directly.

        Call only after any upload for this chunk has already succeeded (or
        after a deliberate dry-run write), so a crash mid-upload can't mark
        work as done that was never actually pushed.
        """
        self._append_csv(self.results_csv_path, rows)
        self._append_csv(self.needs_research_csv_path, needs_research_rows or [])
        self.done_images.update(image_names)
        self._save_manifest()

    def mark_preexisting(self, image_names: set[str]) -> None:
        """Mark IMAGE_NAMES as done without appending any result rows — used
        to seed the checkpoint from images already present in Label Studio
        (see LabelStudioImporter.uploaded_filenames), which have no local
        Glaswacht assessment of their own to record.
        """
        added = image_names - self.done_images
        if not added:
            return
        self.done_images.update(added)
        self._save_manifest()

    def _append_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        model_count = len(self.model_names)
        fieldnames = ["image"] + [
            f"model_{i}_{attr}"
            for i in range(1, model_count + 1)
            for attr in ("rating", "rationale")
        ]
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerows(rows)

    def _save_manifest(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "model_names": self.model_names,
            "done_images": sorted(self.done_images),
        }
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.replace(self.manifest_path)  # atomic rename on POSIX


def load_or_create_checkpoint(
    checkpoint_dir: str | Path, model_names: list[str]
) -> Checkpoint:
    """Load an existing checkpoint from CHECKPOINT_DIR, or start a fresh one.

    Raises ValueError if an existing checkpoint was created with a different
    set of models — resuming with a different model set would silently mix
    incompatible ratings under the same model_N slots.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / MANIFEST_FILENAME

    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_models = data.get("model_names", [])
        if existing_models != model_names:
            raise ValueError(
                f"Checkpoint at {checkpoint_dir} was created with models "
                f"{existing_models}, but this run resolved to {model_names}. "
                "Use the same --model selection to resume, or a different "
                "--checkpoint-dir to start fresh."
            )
        return Checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_names=model_names,
            done_images=set(data.get("done_images", [])),
        )

    return Checkpoint(checkpoint_dir=checkpoint_dir, model_names=model_names)
