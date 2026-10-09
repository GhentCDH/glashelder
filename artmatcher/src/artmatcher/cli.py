"""CLI for matching images from the Glashelder dataset to images from scraped databases (like WikiArt or KIK-IRPA)"""

import json
import logging
from pathlib import Path
from typing import Literal

import click

from artscraper import Artwork
from common import ASSETS_DIR, run_app
from common.cli import options, serve_options
from common.config import config
from embed import EMBEDDING_MODELS

logger = logging.getLogger(__name__)


def load_artworks(path: str | Path | None) -> dict[str, Artwork]:
    """The artwork records a scrape produced, keyed by the reference they describe."""
    if path is None or not Path(path).is_file():
        if path is not None:
            logger.warning(f"no artworks at {path}, results will carry no metadata")
        return {}

    import artscraper

    found = artscraper.index(path)
    logger.info(f"loaded metadata for {len(found)} artwork references from {path}")
    return found


def resolve_subset(path: Path, source, root: str | Path | None) -> list[int]:
    """The glashelder indices of the images a `--subset` file lists.

    A line matches an image by its path relative to `root`, by its full path, or
    by its file name, in that order. Lines that match nothing are reported, so a
    typo or a list for another image directory does not go unnoticed.
    """
    served = list(source.indices)
    by_key: dict[str, int] = {}
    for index in served:
        image = Path(source.glashelder_path(index) or "")
        by_key.setdefault(image.name, index)
        by_key[str(image)] = index
        if root is not None and image.is_relative_to(root):
            by_key[str(image.relative_to(root))] = index

    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    wanted = [line for line in lines if line]
    found = {by_key[line] for line in wanted if line in by_key}
    missing = [line for line in wanted if line not in by_key]

    if not found:
        raise click.ClickException(
            f"none of the {len(wanted):,} images in {path} are among the "
            f"{len(served):,} glashelder images being served"
        )
    logger.info(f"labeling a subset: {len(found):,} images from {path}")
    if missing:
        logger.warning(
            f"{len(missing):,} lines in {path} match no glashelder image, "
            f"for example {missing[:3]}"
        )
    return sorted(found)


@click.group()
def artmatcher():
    """Match glass plate photos to known artworks using feature matching and ML.

    Artifact locations and tuning defaults come from `glashelder.toml`, so the
    options below only need to be given when overriding it.
    """


@artmatcher.command()
@click.argument("dataset", type=click.Path(exists=True), required=True)
@click.argument("output_model", type=click.Path(), required=True)
@click.option(
    "--n-estimators",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees in the Random Forest",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation folds",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel workers for ORB/hash feature extraction (1=sequential)",
)
@click.option(
    "--no-image-features",
    is_flag=True,
    help="Skip ORB/hash feature extraction (faster, but less accurate)",
)
def train_classifier(
    dataset: str,
    output_model: str,
    n_estimators: int,
    cv_folds: int,
    workers: int,
    no_image_features: bool,
):
    """Train an ML classifier from a labeled dataset.

    DATASET is the JSON file exported from the labeling web UI.
    OUTPUT_MODEL is where to save the trained model (.pkl file).

    The dataset should contain labeled examples in the format exported
    by the 'serve' command's "Download Dataset" button.

    Use --workers to parallelize ORB/hash feature extraction across CPU cores.
    For a 96-core machine, try --workers 48 or higher.

    Example:
        glashelder artmatcher train-classifier labeling_dataset.json model.pkl --workers 48
    """
    from .ml_classifier import (
        MLClassifier,
    )

    classifier = MLClassifier()

    try:
        classifier.train(
            dataset,
            n_estimators=n_estimators,
            cv_folds=cv_folds,
            extract_image_features=not no_image_features,
            workers=workers,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    classifier.save(output_model)


REFERENCE_OPTIONS = [
    # where the references come from, shared by `serve` and `batch-classify`
    click.option(
        "--reference-embeddings",
        "-e",
        type=click.Path(exists=True),
        default=None,
        help="Store (or legacy pickle) of the reference artworks' embeddings, "
        "all backends together",
    ),
    click.option(
        "--source",
        "sources",
        multiple=True,
        help="Only match against artworks from this backend, repeatable. "
        "Needs --artworks to know where each reference came from.",
    ),
    click.option(
        "--glashelder-images",
        "-i",
        type=click.Path(exists=True, file_okay=False),
        default=ASSETS_DIR,
        help="Directory the glashelder images live in on this machine",
    ),
]


@artmatcher.command()
@options(*REFERENCE_OPTIONS)
@click.option(
    "--glashelder-embeddings",
    "-g",
    type=str,
    default=None,
    help="Store (or legacy pickle) of the glashelder embeddings; "
    "made on the fly when not given",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(sorted(EMBEDDING_MODELS)),
    help="Embedding model to use",
)
@click.option("--top-k", default=10, help="Number of top matches to consider")
@click.option(
    "--candidates",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="With --ml-model: how many of the top CLIP matches to classify while "
    "the ones before them are no match; the most confident one is shown",
)
@click.option(
    "--threshold", default=0.2, help="Diff threshold for match classification"
)
@click.option(
    "--ml-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to trained ML model (.pkl file). Without it, matches are judged "
    "by CLIP similarity alone",
)
@click.option(
    "--search-workers",
    "-sw",
    type=int,
    default=8,
    show_default=True,
    help="Number of parallel workers for ORB/hash extraction during search (ML classifier only)",
)
@click.option(
    "--search-batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Number of images to check in parallel when searching for matches",
)
@click.option(
    "--results-json",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Path to pre-computed batch_classify JSON output. Skips on-the-fly computation.",
)
@click.option(
    "--reference-images",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Directory where the reference images live. Not needed when the "
    "references are URLs from a scrape.",
)
@click.option(
    "--artworks",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Scraped artworks (JSON or JSONL), used to show each match's title, "
    "artist and source link",
)
@click.option(
    "--session-dir",
    type=click.Path(file_okay=False),
    default="labeling-sessions",
    show_default=True,
    help="Directory where labeling sessions are stored (one JSON file each)",
)
@click.option(
    "--subset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Only label the glashelder images listed in this file: one per line, as "
    "a file name or a path relative to --glashelder-images ('#' starts a "
    "comment). Uses the same stores and saved top-k as the full set.",
)
@serve_options(5001)
def serve(
    reference_embeddings: str | None,
    sources: tuple[str, ...],
    glashelder_images: str | None,
    glashelder_embeddings: str | None,
    model: str,
    top_k: int,
    candidates: int,
    threshold: float,
    ml_model: str | None,
    search_workers: int,
    search_batch_size: int,
    results_json: str | None,
    reference_images: Path | None,
    artworks: str | None,
    session_dir: str,
    subset: Path | None,
    **server,
):
    """Start a web interface for labeling artwork matches.

    Unless --no-auth is given, set the `secret` config value
    (or GLASHELDER_SECRET), then access the server with /?secret=<your-secret> to
    get a session cookie.

    Use --results-json to load pre-computed results from batch_classify.
    The threshold can still be adjusted interactively in the UI.

    Without it, matches are computed on the fly, by the ML classifier when
    --ml-model is given and by CLIP similarity alone otherwise.
    Use --search-workers to parallelize ORB/hash extraction during search.
    """
    if glashelder_images is None:
        raise click.UsageError(
            "no glashelder images: pass --glashelder-images or set "
            "paths.images.glashelder"
        )
    # references scraped as URLs need no local directory
    reference_images = Path(reference_images) if reference_images is not None else None

    logger.info(f"Loading artworks from {artworks}...")
    artwork_index = load_artworks(artworks)

    from .web.app import create_app

    # If results_json is provided, use pre-computed results mode
    if results_json is not None:
        from .batch import read_json_output
        from .web.source import PrecomputedSource

        logger.info(f"Loading pre-computed results from {results_json}...")
        results = read_json_output(results_json)["results"]
        logger.info(f"Loaded {len(results)} pre-computed results")

        source = PrecomputedSource(
            results, glashelder_images, reference_images, artworks=artwork_index
        )
    else:
        from .loading import load_classifier, load_matching_inputs
        from .web.source import LiveSource

        if reference_embeddings is None:
            raise click.UsageError(
                "no references: pass --reference-embeddings or set "
                "paths.embeddings.reference (or use --results-json)"
            )
        # `None` when ml_model isn't given
        ml_classifier_instance = load_classifier(ml_model, threshold)
        try:
            inputs = load_matching_inputs(
                glashelder_embeddings=glashelder_embeddings,
                reference_embeddings=reference_embeddings,
                model=model,
                image_dir=glashelder_images,
                reference_root=reference_images,
                glashelder_root=glashelder_images,
                artworks=artwork_index,
                sources=sources,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

        logger.info("Creating live source from inputs")

        source = LiveSource(
            inputs,
            top_k=top_k,
            classifier=ml_classifier_instance,
            search_workers=search_workers,
            search_batch_size=search_batch_size,
            artworks=artwork_index,
            candidates=candidates,
        )
        # the source kept only the top-k: let the matrices go (and free the VRAM)
        del inputs

    chosen = (
        resolve_subset(subset, source, glashelder_images)
        if subset is not None
        else None
    )
    app = create_app(
        source, threshold=threshold, session_dir=session_dir, subset=chosen
    )
    run_app(app, **server)


@artmatcher.command()
@options(*REFERENCE_OPTIONS)
@click.option(
    "--ml-model",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained ML classifier (.pkl file)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path (.json or .csv)",
)
@click.option(
    "--glashelder-embeddings",
    "-g",
    type=click.Path(exists=True),
    default=None,
    help="Pre-computed glashelder embeddings (generates if not provided)",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv", "both"]),
    default="json",
    show_default=True,
    help="Output format",
)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Parallel workers for ORB extraction",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    show_default=True,
    help="Top reference matches to consider",
)
@click.option(
    "--candidates",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="How many of the top CLIP matches to classify while the ones before "
    "them are no match; the most confident one is kept",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Confidence threshold for is_match",
)
@click.option(
    "--min-confidence",
    type=float,
    default=None,
    help="Only output results above this confidence",
)
@click.option(
    "--cache-features",
    is_flag=True,
    help="Cache features to disk for reuse",
)
@click.option(
    "--cache-path",
    type=click.Path(),
    default="./feature_cache.pkl",
    show_default=True,
    help="Path for feature cache",
)
@click.option(
    "--cache-save-interval",
    type=int,
    default=100,
    show_default=True,
    help="Save cache to disk every N items (0 to disable periodic saves)",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Timeout in seconds per item for feature extraction",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(sorted(EMBEDDING_MODELS)),
    help="Embedding model to use if generating embeddings",
)
@click.option(
    "--artworks",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Scraped artworks (JSON or JSONL), joined onto every result so matches "
    "carry their title, artist and source link",
)
def batch_classify(
    reference_embeddings: str | None,
    sources: tuple[str, ...],
    glashelder_images: str | None,
    ml_model: str,
    output: str,
    glashelder_embeddings: str | None,
    output_format: Literal["json", "csv", "both"],
    workers: int,
    top_k: int,
    candidates: int,
    threshold: float,
    min_confidence: float | None,
    cache_features: bool,
    cache_path: str,
    cache_save_interval: int,
    timeout: float,
    model: str,
    artworks: str | None,
):
    """Batch classify all glashelder images for artwork matches.

    This command processes all glashelder images through the artwork matcher,
    computing embedding similarities and running ML classification with
    parallel ORB/hash feature extraction.

    Example:
        glashelder artmatcher batch-classify \\
            -e artworks-clip.store \\
            --ml-model model.pkl \\
            -o results.json \\
            --workers 32
    """
    if reference_embeddings is None:
        raise click.UsageError(
            "no references: pass --reference-embeddings or set "
            "paths.embeddings.reference"
        )
    from .batch import (
        FeatureCache,
        write_csv_output,
        write_json_output,
    )
    from .batch import (
        batch_classify as do_batch_classify,
    )
    from .loading import load_classifier, load_matching_inputs, top_matches

    classifier = load_classifier(ml_model, threshold)
    assert classifier is not None
    artwork_index = load_artworks(artworks)
    try:
        inputs = load_matching_inputs(
            reference_embeddings=reference_embeddings,
            glashelder_embeddings=glashelder_embeddings,
            model=model,
            image_dir=glashelder_images,
            reference_root=config().paths.images.reference,
            glashelder_root=glashelder_images,
            artworks=artwork_index,
            sources=sources,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Setup feature cache
    cache = FeatureCache(cache_path) if cache_features else None

    # Run batch classification
    similarities, indices = top_matches(inputs, top_k)
    result = do_batch_classify(
        glashelder_paths=inputs.glashelder_paths,
        artwork_paths=inputs.artwork_paths,
        top_indices=indices,
        top_similarities=similarities,
        classifier=classifier,
        top_k=top_k,
        threshold=threshold,
        workers=workers,
        min_confidence=min_confidence,
        cache=cache,
        timeout_per_item=timeout,
        cache_save_interval=cache_save_interval,
        candidates=candidates,
    )

    # Write output
    output_path = Path(output)
    if output_format == "json":
        write_json_output(result, output_path, artworks=artwork_index)
    elif output_format == "csv":
        csv_path = output_path.with_suffix(".csv")
        write_csv_output(result, csv_path, artworks=artwork_index)
    else:  # both
        write_json_output(
            result, output_path.with_suffix(".json"), artworks=artwork_index
        )
        write_csv_output(
            result, output_path.with_suffix(".csv"), artworks=artwork_index
        )

    # Save cache if enabled
    if cache:
        cache.save()

    logger.info("Batch classification complete!")


@artmatcher.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="The `json` file containing the results from batch-classify",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    required=True,
    help="The output directory where plots' pngs will be stored",
)
def plot_results(input, output_dir):
    """Generate analysis plots from batch classification results."""
    from .plot import (
        extract_arrays,
        load_data,
        plot_confidence_distribution,
        plot_feature_correlations,
        plot_hash_orb,
        plot_statistical_analysis,
        setup_style,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input} ...")
    data = load_data(input)
    results = data["results"]
    meta = data.get("metadata", {})
    summary = data.get("summary", {})
    logger.info(f"  {len(results)} results loaded")

    d = extract_arrays(results)
    d["is_match"] = d["is_match"].astype(bool)

    setup_style()

    logger.info("Generating plots ...")
    plot_confidence_distribution(d, meta, summary, output_dir)
    plot_feature_correlations(d, output_dir)
    plot_hash_orb(d, output_dir)
    plot_statistical_analysis(d, output_dir)
    logger.info("Done!")


@artmatcher.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--labels",
    "label_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A labeling session (or a downloaded dataset) to measure accuracy "
    "against, repeatable. A later label of the same image replaces an earlier one.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Count matches at this confidence instead of the run's threshold",
)
@click.option(
    "--artworks",
    type=click.Path(dir_okay=False),
    default=None,
    help="Scraped artworks, to find the source of results written without "
    "artwork metadata",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("report.html"),
    show_default=True,
    help="Where to write the HTML report",
)
def report(
    results: Path,
    label_files: tuple[Path, ...],
    threshold: float | None,
    artworks: str | None,
    output: Path,
):
    """Report how matching went over a whole `batch-classify` run.

    RESULTS is the JSON `batch-classify` wrote. The report counts the matches,
    by source and by CLIP rank, and with --labels estimates how many of them are
    right and how many true matches the run missed. It is one self-contained
    HTML file; a summary is printed here too.
    """
    from .batch import read_json_output
    from .report import build, render, summary_lines
    from .web.store import normalize_dataset

    data = read_json_output(results)
    labels = []
    for path in label_files:
        try:
            labels.append(normalize_dataset(json.loads(path.read_text())))
        except (ValueError, json.JSONDecodeError) as e:
            raise click.ClickException(f"{path} is not a labeling session: {e}") from e

    # only needed when the run did not join the artwork metadata itself
    unjoined = any(not r.get("artwork") for r in data.get("results", []))
    artwork_index = load_artworks(artworks) if unjoined else None

    found = build(
        data, labels, threshold=threshold, artworks=artwork_index, source=str(results)
    )
    output.write_text(render(found))
    for line in summary_lines(found):
        click.echo(line)
    click.echo(f"-> {output}")
