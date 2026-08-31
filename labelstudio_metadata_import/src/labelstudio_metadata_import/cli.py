import json
import logging

import click

from labelstudio_metadata_import import sampling
from labelstudio_metadata_import.dedupe import DedupeCache
from labelstudio_metadata_import.importer import LabelStudioImporter, resolve_image_files
from labelstudio_metadata_import.pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CSV,
    DEFAULT_DEDUPE_WORKERS,
    DEFAULT_INTERJECT_SAFE_PERCENT,
    assess_for_labelstudio,
    import_images,
    import_results,
    run_glaswacht_checkpointed,
)
from labelstudio_metadata_import.sampling import (
    list_images,
    random_sample,
    resolve_sample_size,
    sample_by_group,
)

logger = logging.getLogger(__name__)

_LS_URL_DEFAULT = "http://localhost:8080"

_model_option = click.option(
    "--model",
    "model_names",
    multiple=True,
    help="Glaswacht model(s) to run (lavaguard, qwen3, pixtral). Repeat to select "
    "multiple. Default: all registered models.",
)


def _project_id_option(envvar="LS_PROJECT_ID"):
    return click.option(
        "--project-id",
        required=True,
        type=int,
        envvar=envvar,
        help=f"Label Studio project ID (env: {envvar}).",
    )


_ls_options = [
    click.option(
        "--ls-url",
        default=_LS_URL_DEFAULT,
        envvar="LS_URL",
        show_default=True,
        help="Label Studio base URL.",
    ),
    click.option(
        "--ls-token",
        envvar="LS_TOKEN",
        required=True,
        help="Label Studio API token.",
    ),
    click.option(
        "--tasks-json",
        default="taken.json",
        show_default=True,
        type=click.Path(),
        help="Where to write the generated tasks for inspection.",
    ),
    click.option(
        "--commit/--dry-run",
        default=False,
        help="Actually import into Label Studio. Default: dry-run, only writes --tasks-json.",
    ),
]


def _ls_command(project_id_envvar="LS_PROJECT_ID"):
    """Decorator factory bundling --ls-url/--ls-token/--tasks-json/--commit
    plus a --project-id backed by PROJECT_ID_ENVVAR — pass a distinct envvar
    per command that should target its own Label Studio project.
    """

    def decorator(func):
        for option in reversed(_ls_options):
            func = option(func)
        return _project_id_option(project_id_envvar)(func)

    return decorator


def _report_import(tasks, missing, tasks_json, commit, project_id, skipped_existing=0):
    if skipped_existing:
        click.echo(f"{skipped_existing} image(s) already in the project, skipped.")
    click.echo(f"{len(tasks)} task(s) written to {tasks_json}")
    if missing:
        click.echo(
            f"WARNING: {len(missing)} CSV row(s) had no matching uploaded file:",
            err=True,
        )
        for name in missing[:10]:
            click.echo(f"   - {name}", err=True)
    if commit:
        click.echo(f"Imported into Label Studio project {project_id}.")
    else:
        click.echo(
            f"Dry run — inspect {tasks_json}, then re-run with --commit to import."
        )


@click.group()
def labelstudio_metadata_import():
    """Run Glaswacht (headless) and import the results into Label Studio."""
    pass


@labelstudio_metadata_import.command("health")
@_project_id_option()
@click.option(
    "--ls-url",
    default=_LS_URL_DEFAULT,
    envvar="LS_URL",
    show_default=True,
    help="Label Studio base URL.",
)
@click.option(
    "--ls-token",
    envvar="LS_TOKEN",
    required=True,
    help="Label Studio API token.",
)
def health(project_id, ls_url, ls_token):
    """Check that LS_URL is reachable, LS_TOKEN is valid, and the project is accessible."""
    click.echo(f"LS_URL={ls_url}")
    click.echo(f"LS_PROJECT_ID={project_id}")
    click.echo(f"LS_TOKEN=<{len(ls_token)} chars, not printed>\n")

    importer = LabelStudioImporter(ls_url, ls_token, project_id)
    results = importer.health_check()

    all_ok = True
    for result in results:
        status = "OK" if result.ok else "FAIL"
        all_ok = all_ok and result.ok
        line = f"[{status}] {result.label}"
        if result.detail:
            line += f": {result.detail}"
        click.echo(line)

    if not all_ok:
        raise SystemExit(1)


@labelstudio_metadata_import.command("known-images")
@_project_id_option()
@click.option(
    "--ls-url",
    default=_LS_URL_DEFAULT,
    envvar="LS_URL",
    show_default=True,
    help="Label Studio base URL.",
)
@click.option(
    "--ls-token",
    envvar="LS_TOKEN",
    required=True,
    help="Label Studio API token.",
)
@click.option(
    "--cache-path",
    default="ls_known_images.json",
    show_default=True,
    type=click.Path(),
    help="Where to persist the content-hash cache for this project. Reused "
    "and incrementally extended on later runs (already-seen tasks aren't "
    "re-fetched).",
)
@click.option(
    "--check-path",
    type=click.Path(exists=True),
    default=None,
    help="Optional: a folder or file to check against the cache after "
    "refreshing it, reporting which local images are already in the project.",
)
def known_images(project_id, ls_url, ls_token, cache_path, check_path):
    """Build/refresh a local cache of every image already in a Label Studio
    project, keyed by content hash (and by filename where available).

    This is the expensive step other commands' --dedupe-cache options rely
    on: Label Studio has no lightweight "list tasks" endpoint, so every task
    is fetched in full (each one carrying its embedded image) to hash it.
    Run this deliberately, ahead of time — not as a side effect of a normal
    upload — and reuse --cache-path afterwards; re-running only fetches
    tasks not already in the cache.
    """
    importer = LabelStudioImporter(ls_url, ls_token, project_id)
    cache = DedupeCache.load(cache_path)
    new_count = cache.refresh(importer)
    click.echo(
        f"{cache_path}: {len(cache.seen_task_ids)} task(s) cached "
        f"({new_count} newly hashed this run)."
    )

    if check_path:
        files = resolve_image_files(check_path)
        dupes = [p for p in files if cache.contains_file(p)]
        click.echo(
            f"{len(dupes)}/{len(files)} local image(s) in {check_path} are "
            f"already in project {project_id}."
        )
        if dupes:
            for p in dupes[:20]:
                click.echo(f"   - {p.name}")
            if len(dupes) > 20:
                click.echo(f"   ... and {len(dupes) - 20} more")


@labelstudio_metadata_import.command("assess")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Image folder or single image to assess with Glaswacht.",
)
@_model_option
@click.option(
    "--output-csv",
    default=DEFAULT_CSV,
    show_default=True,
    help="Where to write the Label Studio-ready CSV.",
)
def assess(path, model_names, output_csv):
    """Run Glaswacht (no GUI) on PATH and write a Label Studio-ready CSV.

    Only images where at least one model rated something other than Safe are
    kept; category and majority_vote are dropped. Columns are model_1_rating/
    model_1_rationale, model_2_rating/model_2_rationale, ... numbered in a
    fixed model order (see the README), not category/majority_vote.
    """
    csv_path, kept, skipped = assess_for_labelstudio(path, model_names, output_csv)
    click.echo(
        f"Wrote {csv_path}: {kept} image(s) flagged for review, "
        f"{skipped} skipped (Safe across all models run)."
    )


@labelstudio_metadata_import.command("import")
@click.option(
    "--image-path",
    required=True,
    type=click.Path(exists=True),
    help="Image folder or single image matching the CSV's 'image' column.",
)
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV with metadata; must have an 'image' column with filenames.",
)
@click.option(
    "--dedupe-cache",
    "dedupe_cache_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Cache built by `known-images` — skip rows whose image is already "
    "in the project (by content hash). Read locally, no network call.",
)
@_ls_command()
def import_cmd(
    image_path, csv_path, dedupe_cache_path, project_id, ls_url, ls_token, tasks_json, commit
):
    """Upload images + CSV metadata to Label Studio as tasks."""
    tasks, missing, skipped_existing = import_results(
        image_path,
        csv_path,
        project_id,
        ls_url,
        ls_token,
        tasks_json,
        commit,
        dedupe_cache_path,
    )
    _report_import(tasks, missing, tasks_json, commit, project_id, skipped_existing)


@labelstudio_metadata_import.command("sample")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Folder of images to sample from.",
)
@click.option(
    "--percent",
    type=click.FloatRange(0, 100),
    default=None,
    help="Percentage of the (matched) pool to select in total (0-100), spread "
    "across collections in --group-csv mode. Exactly one of --percent/--count.",
)
@click.option(
    "--count",
    type=click.IntRange(min=0),
    default=None,
    help="Total absolute number of images to select, spread across collections "
    "in --group-csv mode. Exactly one of --percent/--count.",
)
@click.option(
    "--group-csv",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Metadata CSV to spread --count/--percent as evenly as possible across "
    "EACH collection (stratified mode), instead of one flat random draw over "
    "the whole folder. Small and large collections then end up roughly "
    "equally represented, rather than the sample being dominated by the "
    "biggest one.",
)
@click.option(
    "--group-column",
    default=sampling.DEFAULT_GROUP_COLUMN,
    show_default=True,
    help="Column in --group-csv holding the collection name.",
)
@click.option(
    "--id-column",
    default=sampling.DEFAULT_ID_COLUMN,
    show_default=True,
    help="Column in --group-csv identifying the image, matched against filenames in --path.",
)
@_ls_command("LS_SAMPLE_PROJECT_ID")
def sample(
    path,
    percent,
    count,
    group_csv,
    group_column,
    id_column,
    project_id,
    ls_url,
    ls_token,
    tasks_json,
    commit,
):
    """Push a truly random subset of images from PATH straight to Label Studio.

    No Glaswacht run, no metadata — just a random draw of images, for spot
    checks or calibration batches. Uses the OS's CSPRNG (secrets.SystemRandom),
    so the selection is not reproducible and cannot be seeded.

    Without --group-csv: one flat random draw of --count/--percent over every
    image in PATH (the original behaviour). With --group-csv: --count/
    --percent is still a TOTAL, but spread as evenly as possible across every
    collection instead of one flat draw, so the sample isn't dominated by the
    biggest collection.
    """
    if (percent is None) == (count is None):
        raise click.UsageError("Pass exactly one of --percent or --count.")

    images = list_images(path)
    if not images:
        raise click.ClickException(f"No images found in {path}")

    if group_csv:
        groups = sampling.read_id_groups(group_csv, group_column, id_column)
        chosen, reports = sample_by_group(images, groups, count, percent)
        present = [r for r in reports if r.matched > 0]
        for r in present:
            click.echo(
                f"  {r.group}: selected {r.selected} (matched {r.matched}/{r.rows} CSV row(s) to a file)"
            )
        empty = len(reports) - len(present)
        if empty:
            click.echo(f"  ({empty} more collection(s) had no matching file in --path)")
        click.echo(
            f"Selected {len(chosen)} image(s) at random across {len(present)} collection(s) with matches."
        )
    else:
        size = resolve_sample_size(len(images), count, percent)
        chosen = random_sample(images, size)
        click.echo(f"Selected {size}/{len(images)} image(s) at random.")

    if not chosen:
        click.echo("Nothing to import.")
        return

    tasks, missing = import_images(
        chosen, project_id, ls_url, ls_token, tasks_json, commit
    )
    _report_import(tasks, missing, tasks_json, commit, project_id)


@labelstudio_metadata_import.command("pipeline")
@click.option(
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="Image folder or single image to assess and import.",
)
@_model_option
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Run in resumable chunks, checkpointed here. Each chunk (--chunk-size "
    "images, every model) is assessed, its flagged images are uploaded, and "
    "it's recorded as done before moving to the next — so re-running with "
    "the same --checkpoint-dir skips already-done images and resumes where "
    "an interrupted run left off, instead of reassessing everything.",
)
@click.option(
    "--chunk-size",
    type=click.IntRange(min=1),
    default=DEFAULT_CHUNK_SIZE,
    show_default=True,
    help="Images per checkpoint, only used with --checkpoint-dir.",
)
@click.option(
    "--interject-safe-percent",
    type=click.FloatRange(0, 100),
    default=DEFAULT_INTERJECT_SAFE_PERCENT,
    show_default=True,
    help="Percent of each chunk's size to randomly interject as genuinely-Safe "
    "control images alongside the flagged ones, only used with --checkpoint-dir "
    "— so reviewers aren't only ever shown pre-flagged images.",
)
@click.option(
    "--dedupe-cache",
    "dedupe_cache_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Cache built by `known-images` — every pending image is content-"
    "hashed and checked against it before chunks are even formed, so any "
    "image already in the project is excluded up front and never sent to "
    "the VLMs. Read locally, no network call.",
)
@click.option(
    "--dedupe-workers",
    type=click.IntRange(min=1),
    default=DEFAULT_DEDUPE_WORKERS,
    show_default=True,
    help="Threads used to hash pending images against --dedupe-cache in "
    "parallel. That step is I/O-bound (reading every pending image off "
    "disk, often a network share), so this is worth raising on a fast "
    "share/local disk and lowering on a slow or bandwidth-limited one.",
)
@_ls_command()
def pipeline(
    path,
    model_names,
    checkpoint_dir,
    chunk_size,
    interject_safe_percent,
    dedupe_cache_path,
    dedupe_workers,
    project_id,
    ls_url,
    ls_token,
    tasks_json,
    commit,
):
    """Run Glaswacht on PATH, then import the flagged results into Label Studio."""
    if checkpoint_dir:
        summary = run_glaswacht_checkpointed(
            path,
            model_names,
            checkpoint_dir,
            chunk_size,
            project_id,
            ls_url,
            ls_token,
            commit,
            interject_safe_percent,
            dedupe_cache_path,
            dedupe_workers,
        )
        click.echo(
            f"{summary['total_images']} image(s) total, "
            f"{summary['already_done']} already checkpointed, "
            f"{summary['already_in_ls']} already in Label Studio (per "
            "--dedupe-cache, excluded before assessment), "
            f"{summary['processed_this_run']} processed this run across "
            f"{summary['chunks']} chunk(s): {summary['flagged_this_run']} flagged, "
            f"{summary['interjected_safe_this_run']} safe-control interjected, "
            f"{summary['needs_research_this_run']} needing manual research."
        )
        click.echo(f"Checkpoint: {summary['checkpoint_dir']}")
        click.echo(f"Full record of every processed image: {summary['results_csv']}")
        if summary["needs_research_this_run"]:
            click.echo(
                f"Needs manual research (not sent to Label Studio): "
                f"{summary['needs_research_csv']}"
            )
        if not commit:
            click.echo("Dry run — re-run with --commit to actually import.")
        return

    csv_path, kept, skipped = assess_for_labelstudio(path, model_names, DEFAULT_CSV)
    click.echo(
        f"Glaswacht: {kept} image(s) flagged for review, "
        f"{skipped} skipped (Safe across all models run)."
    )
    if kept == 0:
        click.echo("Nothing to import.")
        return
    tasks, missing, skipped_existing = import_results(
        path,
        str(csv_path),
        project_id,
        ls_url,
        ls_token,
        tasks_json,
        commit,
        dedupe_cache_path,
    )
    _report_import(tasks, missing, tasks_json, commit, project_id, skipped_existing)


@labelstudio_metadata_import.command("test-run")
@click.option(
    "--image",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Single image to run through Glaswacht end-to-end.",
)
@_model_option
@click.option(
    "--project-id",
    type=int,
    envvar="LS_PROJECT_ID",
    help="Label Studio project ID. Omit to skip the Label Studio step.",
)
@click.option("--ls-url", default=_LS_URL_DEFAULT, envvar="LS_URL", show_default=True)
@click.option("--ls-token", envvar="LS_TOKEN", help="Label Studio API token.")
@click.option(
    "--commit/--dry-run",
    default=False,
    help="Actually create the task in Label Studio. Default: dry-run.",
)
def test_run(image, model_names, project_id, ls_url, ls_token, commit):
    """Smoke test: run Glaswacht on ONE image and preview (or perform) the
    Label Studio import for it. Meant for checking the whole pipeline works
    before pointing it at a full batch.
    """
    click.echo(f"[1/2] Running Glaswacht on {image} ...")
    csv_path, kept, _skipped = assess_for_labelstudio(image, model_names, DEFAULT_CSV)

    import csv as csv_module

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    click.echo(f"Glaswacht result ({csv_path}):")
    click.echo(json.dumps(rows, indent=2, ensure_ascii=False))

    if kept == 0:
        click.echo("\nSafe across all models run — nothing to send to Label Studio.")
        return

    if not project_id or not ls_token:
        click.echo(
            "\nSkipping Label Studio step: pass --project-id/--ls-token "
            "(or set LS_PROJECT_ID/LS_TOKEN) to test that too."
        )
        return

    click.echo(
        f"\n[2/2] {'Importing into' if commit else 'Dry-run against'} "
        f"Label Studio project {project_id} ..."
    )
    tasks, missing, skipped_existing = import_results(
        image, str(csv_path), project_id, ls_url, ls_token, "taken.json", commit
    )
    click.echo(json.dumps(tasks, indent=2, ensure_ascii=False))
    _report_import(tasks, missing, "taken.json", commit, project_id, skipped_existing)
