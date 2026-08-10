"""Glue between Glaswacht (risk_assessment, headless) and the Label Studio import."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from labelstudio_metadata_import.checkpoint import Checkpoint, load_or_create_checkpoint
from labelstudio_metadata_import.importer import (
    LabelStudioImporter,
    build_tasks_from_rows,
    csv_image_filenames,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV = "labelstudio_review.csv"
DEFAULT_CHUNK_SIZE = 200
SAFE_RATING = "Safe"


def _resolve_model_classes(model_names: tuple[str, ...]) -> tuple[list[str], list]:
    """Resolve --model names to (ordered_names, model_classes), in the
    registry's fixed order (not necessarily the order --model was repeated on
    the CLI) — so "model_1" always refers to the same physical model.
    """
    from risk_assessment.cli import MODEL_REGISTRY

    names = model_names or tuple(MODEL_REGISTRY.keys())
    ordered_names = [n for n in MODEL_REGISTRY if n in names]
    model_classes = [MODEL_REGISTRY[n] for n in ordered_names]
    return ordered_names, model_classes


def run_glaswacht(path: str, model_names: tuple[str, ...] = ()) -> dict[str, list]:
    """Run Glaswacht's safety models on PATH without the TUI.

    Returns {model_name: [Rating, ...]} in a stable order — see
    _resolve_model_classes.

    Note: risk_assessment's own Orchestrator.run_all() always writes the full,
    unfiltered assessment (including category + majority_vote) to
    combined_risk_assessment_results.csv in the current directory as a side
    effect — that's its own archival record and is untouched here.
    """
    from risk_assessment.orchestrator import Orchestrator

    _, model_classes = _resolve_model_classes(model_names)
    orchestrator = Orchestrator(model_classes)
    orchestrator.run_all(path)
    return orchestrator.assessments


def combine_ratings(assessments: dict[str, list]) -> tuple[list[dict], int]:
    """Reduce {model_name: [Rating, ...]} to one row per image with
    model_1_rating/model_1_rationale, model_2_rating/model_2_rationale, ...
    (numbered in the order models were run — see run_glaswacht). No category,
    no majority_vote: those never leave this pipeline.

    Returns (rows, model_count).
    """
    model_names = list(assessments.keys())
    combined: dict[str, dict] = {}
    for i, model_name in enumerate(model_names, start=1):
        for assessment in assessments[model_name]:
            row = combined.setdefault(assessment.image, {"image": assessment.image})
            row[f"model_{i}_rating"] = assessment.rating
            row[f"model_{i}_rationale"] = assessment.rationale
    return list(combined.values()), len(model_names)


def needs_review(row: dict, model_count: int) -> bool:
    """True unless every model that ran rated the image Safe."""
    ratings = [row.get(f"model_{i}_rating") for i in range(1, model_count + 1)]
    return not all(r == SAFE_RATING for r in ratings)


def write_labelstudio_csv(rows: list[dict], output_csv: str, model_count: int) -> Path:
    fieldnames = ["image"] + [
        f"model_{i}_{field}"
        for i in range(1, model_count + 1)
        for field in ("rating", "rationale")
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return Path(output_csv)


def assess_for_labelstudio(
    path: str, model_names: tuple[str, ...] = (), output_csv: str = DEFAULT_CSV
) -> tuple[Path, int, int]:
    """Run Glaswacht on PATH and write a Label Studio-ready CSV containing only
    the images where at least one model rated something other than Safe.

    Returns (csv_path, kept_count, skipped_count).
    """
    assessments = run_glaswacht(path, model_names)
    all_rows, model_count = combine_ratings(assessments)
    kept = [row for row in all_rows if needs_review(row, model_count)]
    write_labelstudio_csv(kept, output_csv, model_count)
    return Path(output_csv), len(kept), len(all_rows) - len(kept)


def import_rows(
    image_path: str | Path | list[str | Path],
    rows: list[dict],
    project_id: int,
    ls_url: str,
    ls_token: str,
    tasks_json: str = "taken.json",
    commit: bool = False,
) -> tuple[list[dict], list[str]]:
    """Upload the images referenced in ROWS (each needs an "image" filename
    key) and their metadata to a Label Studio project. IMAGE_PATH is a
    folder, single file, or explicit list of file paths — see
    LabelStudioImporter.upload_images. Always writes TASKS_JSON for
    inspection; only calls the import API when COMMIT is set.
    """
    if not rows:
        Path(tasks_json).write_text("[]", encoding="utf-8")
        return [], []

    filenames = {row["image"] for row in rows}
    importer = LabelStudioImporter(ls_url, ls_token, project_id)
    server_paths = importer.upload_images(image_path, filenames=filenames)
    tasks, missing = build_tasks_from_rows(rows, server_paths)

    Path(tasks_json).write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if commit:
        result = importer.import_tasks(tasks)
        logger.info("Imported into Label Studio: %s", result)

    return tasks, missing


def import_results(
    image_path: str,
    csv_path: str,
    project_id: int,
    ls_url: str,
    ls_token: str,
    tasks_json: str = "taken.json",
    commit: bool = False,
) -> tuple[list[dict], list[str]]:
    """Upload the images referenced in CSV_PATH and its metadata to a Label
    Studio project. Only images actually listed in the CSV are uploaded — if
    IMAGE_PATH is a folder with more (e.g. Safe-rated) images than the CSV,
    those are skipped. Always writes TASKS_JSON for inspection; only calls the
    import API when COMMIT is set.
    """
    filenames = csv_image_filenames(csv_path)
    if not filenames:
        Path(tasks_json).write_text("[]", encoding="utf-8")
        return [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return import_rows(
        image_path, rows, project_id, ls_url, ls_token, tasks_json, commit
    )


def import_images(
    image_paths: list[Path],
    project_id: int,
    ls_url: str,
    ls_token: str,
    tasks_json: str = "taken.json",
    commit: bool = False,
) -> tuple[list[dict], list[str]]:
    """Upload IMAGE_PATHS as-is and create one plain {"image": ...} task per
    image — no Glaswacht run, no metadata CSV. Used by random sampling, where
    there's no per-image rating to attach.
    """
    if not image_paths:
        Path(tasks_json).write_text("[]", encoding="utf-8")
        return [], []

    folder = image_paths[0].parent
    filenames = {p.name for p in image_paths}

    importer = LabelStudioImporter(ls_url, ls_token, project_id)
    server_paths = importer.upload_images(folder, filenames=filenames)

    tasks, missing = [], []
    for p in image_paths:
        if p.name not in server_paths:
            missing.append(p.name)
            continue
        tasks.append({"data": {"image": server_paths[p.name]}})

    Path(tasks_json).write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if commit:
        result = importer.import_tasks(tasks)
        logger.info("Imported into Label Studio: %s", result)

    return tasks, missing


def run_glaswacht_checkpointed(
    path: str,
    model_names: tuple[str, ...],
    checkpoint_dir: str,
    chunk_size: int,
    project_id: int | None,
    ls_url: str,
    ls_token: str | None,
    commit: bool,
) -> dict:
    """Run Glaswacht over PATH in resumable chunks of CHUNK_SIZE images.

    After every chunk (every configured model has rated every image in it),
    the chunk's flagged rows are uploaded to Label Studio — if PROJECT_ID and
    LS_TOKEN are given — and the chunk is recorded as done in CHECKPOINT_DIR
    before moving on to the next one. Re-running with the same
    CHECKPOINT_DIR skips images already recorded as done, so an interrupted
    run only redoes its current, in-flight chunk, not the whole batch.

    A chunk is the unit of both crash-safety and "done" tracking: if a model
    fails to rate a specific image within a chunk (corrupt file, decode
    error — see risk_assessment's per-image skip handling), that image still
    counts as done once its chunk finishes, just with missing model_N_rating
    fields; needs_review() treats a missing rating as "needs review" rather
    than silently dropping it, and retrying it forever isn't an option that
    scales (a permanently-corrupt file would never let the run finish).

    Returns a summary dict: total_images, already_done, processed_this_run,
    flagged_this_run, chunks, checkpoint_dir, results_csv.
    """
    from risk_assessment.models.utils import get_images
    from risk_assessment.orchestrator import Orchestrator

    ordered_names, model_classes = _resolve_model_classes(model_names)
    ck: Checkpoint = load_or_create_checkpoint(checkpoint_dir, ordered_names)

    all_images = get_images(path)
    pending = [f for f in all_images if os.path.basename(f) not in ck.done_images]
    already_done = len(all_images) - len(pending)
    logger.info(
        "%d image(s) total, %d already checkpointed, %d pending",
        len(all_images),
        already_done,
        len(pending),
    )

    chunks = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
    orchestrator = Orchestrator(model_classes)

    processed_this_run = 0
    flagged_this_run = 0
    for chunk_num, chunk in enumerate(chunks, start=1):
        logger.info(
            "Chunk %d/%d: processing %d image(s)...", chunk_num, len(chunks), len(chunk)
        )
        chunk_assessments = orchestrator.run_chunk(chunk)
        chunk_rows, model_count = combine_ratings(chunk_assessments)
        flagged = [row for row in chunk_rows if needs_review(row, model_count)]

        if project_id is not None and ls_token:
            tasks_json = str(Path(checkpoint_dir) / f"chunk_{chunk_num:04d}_tasks.json")
            tasks, missing = import_rows(
                chunk, flagged, project_id, ls_url, ls_token, tasks_json, commit
            )
            logger.info(
                "Chunk %d/%d: %d/%d flagged, %d task(s) %s%s",
                chunk_num,
                len(chunks),
                len(flagged),
                len(chunk_rows),
                len(tasks),
                "imported" if commit else "dry-run written",
                f", {len(missing)} missing" if missing else "",
            )
        else:
            logger.info(
                "Chunk %d/%d: %d/%d flagged (no --project-id/--ls-token — "
                "assessed only, not uploaded)",
                chunk_num,
                len(chunks),
                len(flagged),
                len(chunk_rows),
            )

        ck.mark_chunk_done([os.path.basename(f) for f in chunk], chunk_rows)
        processed_this_run += len(chunk)
        flagged_this_run += len(flagged)

    return {
        "total_images": len(all_images),
        "already_done": already_done,
        "processed_this_run": processed_this_run,
        "flagged_this_run": flagged_this_run,
        "chunks": len(chunks),
        "checkpoint_dir": str(checkpoint_dir),
        "results_csv": str(ck.results_csv_path),
    }
