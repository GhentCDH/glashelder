"""Glue between Glaswacht (risk_assessment, headless) and the Label Studio import."""

from __future__ import annotations

import csv
import json
import logging
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

from labelstudio_metadata_import.checkpoint import Checkpoint, load_or_create_checkpoint
from labelstudio_metadata_import.importer import (
    LabelStudioImporter,
    build_tasks_from_paths,
    resolve_image_files_by_name,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV = "labelstudio_review.csv"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_INTERJECT_SAFE_PERCENT = 2.0
DEFAULT_DEDUPE_WORKERS = 16
SAFE_RATING = "Safe"
POTENTIALLY_PROBLEMATIC_RATING = "Potentially Problematic"
UNSAFE_RATING = "Unsafe"
VALID_RATINGS = {SAFE_RATING, POTENTIALLY_PROBLEMATIC_RATING, UNSAFE_RATING}


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


def classify_rows(
    rows: list[dict], model_count: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split ROWS into (flagged, safe, needs_research), used by the
    checkpointed pipeline instead of the simpler needs_review():

    - needs_research: at least one model returned a rating outside
      {Safe, Potentially Problematic, Unsafe} — e.g. "Unknown"/"Error" from
      risk_assessment failing to parse that model's output. That's not a
      real safety judgment, so these are never sent to Label Studio; they
      need a human to look at the image (and/or the raw model output)
      directly.
    - safe: every model that ran rated exactly Safe (and every rating is
      valid — a row is never in both buckets).
    - flagged: everything else — at least one model flagged it, and every
      rating is valid.
    """
    flagged, safe, needs_research = [], [], []
    for row in rows:
        ratings = [row.get(f"model_{i}_rating") for i in range(1, model_count + 1)]
        if any(r not in VALID_RATINGS for r in ratings):
            needs_research.append(row)
        elif all(r == SAFE_RATING for r in ratings):
            safe.append(row)
        else:
            flagged.append(row)
    return flagged, safe, needs_research


def interject_safe_rows(
    safe_rows: list[dict], chunk_size: int, percent: float
) -> list[dict]:
    """Randomly select PERCENT% of CHUNK_SIZE from SAFE_ROWS (capped by how
    many are actually available) to mix into the upload as controls.

    Every image sent to Label Studio otherwise passed through Glaswacht's
    Safe filter first — reviewers only ever see images the models already
    flagged. Interjecting a few genuinely-Safe images (indistinguishable
    from the flagged ones in the review UI) gives a baseline read on the
    models' true negative rate and keeps reviewers from assuming everything
    in front of them is already suspect.

    Uses secrets.SystemRandom (not the default seeded PRNG), same as the
    `sample` command — the selection is intentionally not reproducible.
    """
    size = min(len(safe_rows), round(chunk_size * percent / 100))
    if size <= 0:
        return []
    return secrets.SystemRandom().sample(safe_rows, size)


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
    """Build tasks for ROWS (each needs an "image" filename key) with their
    image embedded as base64 — see build_tasks_from_paths for why a plain
    upload-then-reference doesn't work for a project with per-model rating
    fields — and import them to a Label Studio project. IMAGE_PATH is a
    folder, single file, or explicit list of file paths. Always writes
    TASKS_JSON for inspection; only calls the import API when COMMIT is set.
    """
    if not rows:
        Path(tasks_json).write_text("[]", encoding="utf-8")
        return [], []

    tasks, missing = build_tasks_from_paths(rows, image_path)

    Path(tasks_json).write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if commit:
        importer = LabelStudioImporter(ls_url, ls_token, project_id)
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
    dedupe_cache_path: str | Path | None = None,
) -> tuple[list[dict], list[str], int]:
    """Import the images referenced in CSV_PATH and their metadata to a Label
    Studio project. Only images actually listed in the CSV are used — if
    IMAGE_PATH is a folder with more (e.g. Safe-rated) images than the CSV,
    those are skipped.

    If DEDUPE_CACHE_PATH is given, rows whose local image content-hashes to
    something already in that cache (built by `known-images` — see
    dedupe.py) are skipped too, so re-running against the same CSV, or a CSV
    covering images from an earlier run, doesn't duplicate tasks. This is a
    local file read, not a network call — build/refresh the cache
    separately with `known-images`, which is the part that's actually
    expensive (Label Studio has no lightweight "list filenames" endpoint;
    every task fetch carries its full embedded image).

    Always writes TASKS_JSON for inspection; only calls the import API when
    COMMIT is set.

    Returns (tasks, missing, skipped_existing_count).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    skipped_existing: list[dict] = []
    if dedupe_cache_path is not None:
        from labelstudio_metadata_import.dedupe import DedupeCache

        cache = DedupeCache.load(dedupe_cache_path)
        resolved = resolve_image_files_by_name(
            image_path, {r["image"] for r in rows if r.get("image")}
        )
        kept = []
        for row in rows:
            local_path = resolved.get(row.get("image"))
            if local_path is not None and cache.contains_file(local_path):
                skipped_existing.append(row)
            else:
                kept.append(row)
        rows = kept

    tasks, missing = import_rows(
        image_path, rows, project_id, ls_url, ls_token, tasks_json, commit
    )
    return tasks, missing, len(skipped_existing)


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
    interject_safe_percent: float = DEFAULT_INTERJECT_SAFE_PERCENT,
    dedupe_cache_path: str | Path | None = None,
    dedupe_workers: int = DEFAULT_DEDUPE_WORKERS,
) -> dict:
    """Run Glaswacht over PATH in resumable chunks of CHUNK_SIZE images.

    After every chunk (every configured model has rated every image in it),
    each row is classified — see classify_rows() — into flagged, safe, or
    needs_research. The upload to Label Studio (if PROJECT_ID and LS_TOKEN
    are given) is the flagged rows plus a small random interjection of safe
    ones (see interject_safe_rows(), INTERJECT_SAFE_PERCENT of the chunk
    size); needs_research rows are never uploaded — see classify_rows(). The
    chunk is then recorded as done in CHECKPOINT_DIR before moving on to the
    next one. Re-running with the same CHECKPOINT_DIR skips images already
    recorded as done, so an interrupted run only redoes its current,
    in-flight chunk, not the whole batch.

    A chunk is the unit of both crash-safety and "done" tracking: if one
    image within an otherwise-successful chunk fails assessment outright
    (corrupt file, decode error — see risk_assessment's per-image skip
    handling), it still counts as done once its chunk finishes, just with
    missing model_N_rating fields — classify_rows() treats that the same as
    an invalid rating (needs_research) rather than silently dropping it, and
    retrying it forever isn't an option that scales (a permanently-corrupt
    file would never let the run finish).

    DEDUPE_CACHE_PATH, if given, is a cache built by the `known-images` CLI
    command (see dedupe.py) — a local file, so consulting it costs no
    network round-trip. Every pending image's content is hashed and checked
    against it *before* chunks are even formed, i.e. before anything is sent
    to Glaswacht: a match is dropped from PENDING right there, marked done
    in the checkpoint (see Checkpoint.mark_preexisting), and so never
    appears in any chunk or reaches the VLMs at all. This is strictly a
    per-image filter — a chunk containing one known image and nine new ones
    still has those nine new ones assessed normally, just without the known
    one alongside them.

    That hashing reads every pending image off disk (often a network share),
    so it's I/O-bound — DEDUPE_WORKERS files are hashed concurrently via a
    thread pool (each thread releases the GIL while blocked on the read, and
    hashlib releases it during the hash itself, so this isn't fighting
    Python's single-core limit the way CPU-bound work would). Only affects
    this local pre-filtering step; the VLM assessment itself is untouched.

    Returns a summary dict: total_images, already_done, already_in_ls,
    processed_this_run, flagged_this_run, interjected_safe_this_run,
    needs_research_this_run, chunks, checkpoint_dir, results_csv,
    needs_research_csv.
    """
    from risk_assessment.models.utils import get_images
    from risk_assessment.orchestrator import Orchestrator

    ordered_names, model_classes = _resolve_model_classes(model_names)
    ck: Checkpoint = load_or_create_checkpoint(checkpoint_dir, ordered_names)

    all_images = get_images(path)
    pending = [f for f in all_images if os.path.basename(f) not in ck.done_images]
    already_done = len(all_images) - len(pending)

    already_in_ls = 0
    if dedupe_cache_path is not None:
        from labelstudio_metadata_import.dedupe import DedupeCache

        dedupe_cache = DedupeCache.load(dedupe_cache_path)
        still_pending, known = [], []
        paths = [Path(f) for f in pending]
        with click.progressbar(
            length=len(pending), label="Checking dedupe cache", show_pos=True
        ) as bar:
            with ThreadPoolExecutor(max_workers=dedupe_workers) as pool:
                for f, is_known in zip(pending, pool.map(dedupe_cache.contains_file, paths)):
                    (known if is_known else still_pending).append(f)
                    bar.update(1)
        if known:
            ck.mark_preexisting({os.path.basename(f) for f in known})
            already_in_ls = len(known)
            logger.info(
                "%d pending image(s) already in Label Studio (per --dedupe-cache) "
                "— excluded before assessment, never sent to the VLMs",
                already_in_ls,
            )
        pending = still_pending

    logger.info(
        "%d image(s) total, %d already checkpointed, %d already in Label Studio, "
        "%d pending",
        len(all_images),
        already_done,
        already_in_ls,
        len(pending),
    )

    chunks = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
    orchestrator = Orchestrator(model_classes)

    processed_this_run = 0
    flagged_this_run = 0
    interjected_this_run = 0
    needs_research_this_run = 0
    for chunk_num, chunk in enumerate(chunks, start=1):
        logger.info(
            "Chunk %d/%d: processing %d image(s)...", chunk_num, len(chunks), len(chunk)
        )
        chunk_assessments = orchestrator.run_chunk(chunk)
        chunk_rows, model_count = combine_ratings(chunk_assessments)
        flagged, safe, needs_research = classify_rows(chunk_rows, model_count)
        interjected = interject_safe_rows(safe, len(chunk), interject_safe_percent)
        to_upload = flagged + interjected

        if project_id is not None and ls_token:
            tasks_json = str(Path(checkpoint_dir) / f"chunk_{chunk_num:04d}_tasks.json")
            tasks, missing = import_rows(
                chunk, to_upload, project_id, ls_url, ls_token, tasks_json, commit
            )
            logger.info(
                "Chunk %d/%d: %d flagged + %d safe-control = %d task(s) %s%s, "
                "%d needing manual research",
                chunk_num,
                len(chunks),
                len(flagged),
                len(interjected),
                len(tasks),
                "imported" if commit else "dry-run written",
                f", {len(missing)} missing" if missing else "",
                len(needs_research),
            )
        else:
            logger.info(
                "Chunk %d/%d: %d flagged + %d safe-control (no --project-id/"
                "--ls-token — assessed only, not uploaded), %d needing manual research",
                chunk_num,
                len(chunks),
                len(flagged),
                len(interjected),
                len(needs_research),
            )

        ck.mark_chunk_done(
            [os.path.basename(f) for f in chunk], chunk_rows, needs_research
        )
        processed_this_run += len(chunk)
        flagged_this_run += len(flagged)
        interjected_this_run += len(interjected)
        needs_research_this_run += len(needs_research)

    return {
        "total_images": len(all_images),
        "already_done": already_done,
        "processed_this_run": processed_this_run,
        "flagged_this_run": flagged_this_run,
        "interjected_safe_this_run": interjected_this_run,
        "needs_research_this_run": needs_research_this_run,
        "already_in_ls": already_in_ls,
        "chunks": len(chunks),
        "checkpoint_dir": str(checkpoint_dir),
        "results_csv": str(ck.results_csv_path),
        "needs_research_csv": str(ck.needs_research_csv_path),
    }
