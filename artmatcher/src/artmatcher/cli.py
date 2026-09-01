"""CLI for matching images from the Glashelder dataset to images from scraped databases (like WikiArt or KIK-IRPA)"""

import json
import logging
from pathlib import Path
from typing import Literal

import click

from common import ASSETS_DIR
from common.config import config, defaults_for

logger = logging.getLogger(__name__)


def load_artworks(path: str | Path | None) -> dict:
    """The artwork records a scrape produced, keyed by the reference they describe."""
    if path is None or not Path(path).is_file():
        if path is not None:
            logger.warning(f"no artworks at {path}, results will carry no metadata")
        return {}

    import artscraper

    found = artscraper.index(path)
    logger.info(f"loaded metadata for {len(found)} artwork references from {path}")
    return found


@click.group()
@click.pass_context
def artmatcher(ctx: click.Context):
    """Match glass plate photos to known artworks using feature matching and ML.

    Artifact locations and tuning defaults come from `glashelder.toml`, so the
    options below only need to be given when overriding it.
    """
    settings = config()
    paths = {
        "artworks": settings.paths.artworks,
        "image_dir": settings.paths.images.glashelder,
        "glashelder_embeddings": settings.paths.embeddings.glashelder,
        "wikiart_embeddings": settings.paths.embeddings.reference,
        "ml_model": settings.paths.ml_model,
    }
    # an unset path must not shadow the default declared on the command itself
    supplied = {
        **defaults_for(settings.artmatcher),
        **{name: str(value) for name, value in paths.items() if value is not None},
    }
    ctx.default_map = dict.fromkeys(("serve", "batch-classify"), supplied)


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
        click.echo(f"Error: {e}", err=True)
        return

    classifier.save(output_model)


@artmatcher.command()
@click.option(
    "--wikiart-embeddings",
    "-w",
    type=str,
    default=None,
    help="Path to the wikiart embeddings '.pkl' file (not needed with --results-json)",
)
@click.option(
    "--glashelder-embeddings",
    "-g",
    type=str,
    default=None,
    help="Path to pre-calculated glashelder embeddings '.pkl' file (optional)",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use",
)
@click.option(
    "--image-dir",
    "-i",
    type=str,
    default=ASSETS_DIR,
    help="Path to the Glashelder images",
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--top-k", default=10, help="Number of top matches to consider")
@click.option(
    "--threshold", default=0.2, help="Diff threshold for match classification"
)
@click.option(
    "--ml-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to trained ML model (.pkl file). Required if --classifier=ml",
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
    "--glashelder-images",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Path of the directory where the glashelder images live",
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
    "--auth", is_flag=True, help="Enable secret url parameter based authentication."
)
@click.option(
    "--debug", is_flag=True, help="Enable the Flask debugger and auto-reloader."
)
@click.option(
    "--session-dir",
    type=click.Path(file_okay=False),
    default="labeling-sessions",
    show_default=True,
    help="Directory where labeling sessions are stored (one JSON file each)",
)
def serve(
    wikiart_embeddings: str,
    glashelder_embeddings: str | None,
    model: Literal["clip", "dinov2", "qwen3vl"],
    image_dir: str,
    host: str,
    port: int,
    top_k: int,
    threshold: float,
    ml_model: str | None,
    search_workers: int,
    search_batch_size: int,
    results_json: str | None,
    glashelder_images: Path | None,
    reference_images: Path | None,
    artworks: str | None,
    auth: bool,
    debug: bool,
    session_dir: str,
):
    """Start a web interface for labeling artwork matches.

    Authentication is required when `--auth` is set. Set the GLASHELDER_SECRET environment variable,
    then access the server with /?secret=<your-secret> to get a session cookie.

    Use --results-json to load pre-computed results from batch_classify.
    The threshold can still be adjusted interactively in the UI.

    Or use --classifier=ml with --ml-model to compute on the fly.
    Use --search-workers to parallelize ORB/hash extraction during search.
    """
    if glashelder_images is None:
        logger.error(
            "!! You need to set the `--glashelder-images` argument to the directory where the glashelder images are on your system"
        )
        return
    else:
        glashelder_images = Path(glashelder_images)

    # references scraped as URLs need no local directory
    reference_images = Path(reference_images) if reference_images is not None else None

    artwork_index = load_artworks(artworks)

    from .web.app import create_app

    # If results_json is provided, use pre-computed results mode
    if results_json is not None:
        from .web.source import PrecomputedSource

        logger.info(f"Loading pre-computed results from {results_json}...")
        with open(results_json) as f:
            precomputed_data = json.load(f)

        results = precomputed_data["results"]
        logger.info(f"Loaded {len(results)} pre-computed results")

        source = PrecomputedSource(
            results, glashelder_images, reference_images, artworks=artwork_index
        )
    else:
        from .loading import load_classifier, load_matching_inputs
        from .web.source import LiveSource

        # `None` when ml_model isn't given
        ml_classifier_instance = load_classifier(ml_model, threshold)
        try:
            inputs = load_matching_inputs(
                glashelder_embeddings=glashelder_embeddings,
                wikiart_embeddings=wikiart_embeddings,
                model=model,
                image_dir=image_dir,
                wikiart_root=reference_images,
                glashelder_root=glashelder_images,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            return

        source = LiveSource(
            inputs,
            top_k=top_k,
            classifier=ml_classifier_instance,
            search_workers=search_workers,
            search_batch_size=search_batch_size,
            artworks=artwork_index,
        )

    app = create_app(source, threshold=threshold, auth=auth, session_dir=session_dir)

    logger.info(f"Starting Flask server on {host}:{port}...")
    logger.info(f"Open http://localhost:{port} in your browser")
    app.run(host=host, port=port, debug=debug)


@artmatcher.command()
@click.option(
    "--wikiart-embeddings",
    "-w",
    type=click.Path(exists=True),
    required=True,
    help="Path to wikiart embeddings pickle file",
)
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
    "--image-dir",
    "-i",
    type=str,
    default=ASSETS_DIR,
    help="Path to glashelder images",
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
    help="Top wikiart matches to consider",
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
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
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
    wikiart_embeddings: str,
    ml_model: str,
    output: str,
    glashelder_embeddings: str | None,
    image_dir: str,
    output_format: Literal["json", "csv", "both"],
    workers: int,
    top_k: int,
    threshold: float,
    min_confidence: float | None,
    cache_features: bool,
    cache_path: str,
    cache_save_interval: int,
    timeout: float,
    model: Literal["clip", "dinov2", "qwen3vl"],
    artworks: str | None,
):
    """Batch classify all glashelder images for artwork matches.

    This command processes all glashelder images through the artwork matcher,
    computing embedding similarities and running ML classification with
    parallel ORB/hash feature extraction.

    Example:
        glashelder artmatcher batch-classify \\
            -w wikiart.pkl \\
            --ml-model model.pkl \\
            -o results.json \\
            --workers 32
    """
    from .batch import (
        FeatureCache,
        write_csv_output,
        write_json_output,
    )
    from .batch import (
        batch_classify as do_batch_classify,
    )
    from .loading import load_classifier, load_matching_inputs

    classifier = load_classifier(ml_model, threshold)
    assert classifier is not None
    artwork_index = load_artworks(artworks)
    try:
        inputs = load_matching_inputs(
            wikiart_embeddings=wikiart_embeddings,
            glashelder_embeddings=glashelder_embeddings,
            model=model,
            image_dir=image_dir,
            wikiart_root=config().paths.images.reference,
            glashelder_root=image_dir,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Setup feature cache
    cache = FeatureCache(cache_path) if cache_features else None

    # Run batch classification
    result = do_batch_classify(
        glashelder_emb=inputs.glashelder_emb,
        glashelder_paths=inputs.glashelder_paths,
        wikiart_emb=inputs.wikiart_emb,
        wikiart_paths=inputs.wikiart_paths,
        classifier=classifier,
        top_k=top_k,
        threshold=threshold,
        workers=workers,
        min_confidence=min_confidence,
        cache=cache,
        timeout_per_item=timeout,
        cache_save_interval=cache_save_interval,
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
    type=str,
    help="The `json` file containing the results from batch-classify",
)
@click.option(
    "--output-dir",
    "-o",
    type=str,
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
