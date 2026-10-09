import logging
from pathlib import Path

# installed packages
import click
from dotenv import load_dotenv

# local source
from common import ASSETS_DIR, PROJECT_ROOT, run_app
from common.cli import serve_options
from embed import EMBEDDING_MODELS

# TODO: put some default images in `./images/` as exampels
DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
CLUSTERS_PATH = "glashelder.clusters"
PACKAGE_ROOT = PROJECT_ROOT / "cluster"
EMBEDDINGS_PATH = PROJECT_ROOT / "embed" / "embeddings.pkl"

logger = logging.getLogger(__name__)


@click.group()
def cluster():
    """Cluster images using embedding similarity."""
    load_dotenv()
    pass


@cluster.command(
    name="cluster",
    help="Cluster images using UMAP dimensionality reduction and HDBSCAN. "
    "The defaults of the UMAP and HDBSCAN options come from [cluster] in glashelder.toml.",
)
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="Path of the embeddings pkl file"
)
@click.option(
    "--image-dir",
    default=ASSETS_DIR,
    help="Directory the embedded images live in on this machine",
)
@click.option(
    "-s",
    "--output",
    type=click.Path(path_type=Path),
    default=CLUSTERS_PATH,
    help="Directory the clustering result is saved in",
)
@click.option(
    "-o",
    "--output-files",
    is_flag=True,
    help="Whether or not to write example images to files",
)
@click.option("--count", default=40000, help="How many points to log to rerun")
@click.option(
    "-r",
    "--reduce",
    is_flag=True,
    help="Whether or not to stop after reducing dimensions",
)
@click.option(
    "-n",
    "--n-neighbors",
    default=30,
    show_default=True,
    help="UMAP `n_neighbors`: higher keeps more global structure",
)
@click.option(
    "-d",
    "--min-dist",
    default=0.0,
    show_default=True,
    help="UMAP `min_dist`: 0 packs similar images tightly",
)
@click.option(
    "--dimensions",
    default=30,
    show_default=True,
    help="Dimensions UMAP reduces to, for HDBSCAN",
)
@click.option(
    "-m",
    "--min-cluster-size",
    default=30,
    show_default=True,
    help="HDBSCAN `min_cluster_size`: smallest group that counts as a cluster",
)
@click.option(
    "--min-samples",
    type=int,
    default=None,
    help="HDBSCAN `min_samples`: higher makes more noise [default: min-cluster-size]",
)
@click.option(
    "--soft/--no-soft",
    default=True,
    show_default=True,
    help="Also save soft clustering, so `cluster serve` can move noise into clusters",
)
@click.option(
    "--soft-top",
    default=3,
    show_default=True,
    help="Strongest clusters saved per image, with their scores",
)
def do_clustering(
    embeddings,
    image_dir,
    output,
    output_files,
    count,
    reduce,
    n_neighbors,
    min_dist,
    dimensions,
    min_cluster_size,
    min_samples,
    soft,
    soft_top,
):
    import os

    from .pipeline import (
        ClusterOptions,
        export_cluster_samples,
        log_cluster_sizes,
        log_to_rerun,
        reduce_only,
        run,
        save_clustering,
    )
    from .rr import init_rerun

    init_rerun(os.getenv("RERUN_ADDR"))

    options = ClusterOptions(
        embeddings=embeddings,
        image_dir=image_dir,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        dimensions=dimensions,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        soft=soft,
        soft_top=soft_top,
    )

    if reduce:
        reduce_only(options, count=20000)
        return

    clustering, image_paths = run(options)
    save_clustering(output, clustering)
    log_to_rerun(clustering, image_paths, count)
    log_cluster_sizes(clustering)

    if output_files:
        export_cluster_samples(
            clustering, image_paths, Path(CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR)
        )


@cluster.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--embeddings",
    type=click.Path(path_type=Path),
    default=None,
    help="Embedding store to check against (default: the one it was made from)",
)
def info(path: Path | None, embeddings: Path | None):
    """Show what a saved clustering holds, and whether it still matches its embeddings."""
    from collections import Counter

    from embed import Store, is_store

    from .result import read_labels, read_meta, status

    if path is None:
        raise click.UsageError("give a clustering result, or set paths.clusters")
    meta = read_meta(path)
    sizes = Counter(read_labels(path).tolist())
    noise = sizes.pop(-1, 0)
    rows = meta["rows"]

    store_path = embeddings or Path(meta["embeddings"])
    state = (
        f"{status(meta, Store.open(store_path))} ({store_path})"
        if is_store(store_path)
        else f"unknown, {store_path} is not an embedding store here"
    )
    umap, hdbscan = meta["umap"], meta["hdbscan"]
    for name, value in (
        ("path", path),
        ("created", meta["created"]),
        ("model", f"{meta['model']} ({meta['model_id']})"),
        ("rows", f"{rows:,}"),
        (
            "umap",
            f"{umap['dimensions']} dims, n_neighbors {umap['n_neighbors']}, "
            f"min_dist {umap['min_dist']}",
        ),
        (
            "hdbscan",
            f"min_cluster_size {hdbscan['min_cluster_size']}, "
            f"min_samples {hdbscan.get('min_samples') or 'unset'}",
        ),
        (
            "soft",
            f"top {soft['top']} per image" if (soft := meta.get("soft")) else "off",
        ),
        ("clusters", f"{len(sizes):,}"),
        ("noise", f"{noise:,} ({noise / rows:.1%})" if rows else "0"),
        (
            "largest",
            ", ".join(f"{label} ({n:,})" for label, n in sizes.most_common(10)) or "-",
        ),
        ("status", state),
    ):
        click.echo(f"{name + ':':10}{value}")


@cluster.command()
@click.option(
    "--clusters",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Clustering result to show (default: paths.clusters)",
)
@click.option(
    "--image-dir",
    default=None,
    help="Directory the clustered images live in on this machine",
)
@serve_options(5002)
def serve(clusters: Path | None, image_dir: str | None, **server):
    """Show a clustering result in the browser, and give its clusters names and groups.

    Names and groups are saved in the result, in `annotations.json`.
    """
    from .web import create_app

    if clusters is None:
        raise click.UsageError("give --clusters, or set paths.clusters")
    run_app(create_app(clusters, image_dir), **server)


@cluster.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    default=None,
    help="Directory the clustered images live in on this machine",
)
@click.option("--samples", default=4, help="Sample images per cluster")
@click.option("--open", "open_", is_flag=True, help="Open the PDF when it is done")
def report(path: Path | None, image_dir: str | None, samples: int, open_: bool):
    """Write a PDF report of a clustering result into it, as `report.pdf`."""
    from .report import build

    if path is None:
        raise click.UsageError("give a clustering result, or set paths.clusters")
    pdf = build(path, image_dir, count=samples)
    logger.info(f"Saved the report to {pdf}")
    if open_:
        click.launch(str(pdf))


@cluster.command()
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="Path to embeddings pickle file"
)
@click.option("--sample-size", default=100000, help="Amount of images to cluster")
@click.option("--batch-size", default=64, help="Batch size for embeddings generation")
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save analysis results (default: cluster_analysis_YYYYMMDD_HHMMSS)",
)
@click.option("--min-cluster-size", default=50, help="Minimum cluster size for HDBSCAN")
@click.option(
    "--hierarchical",
    is_flag=True,
    help="Enable hierarchical refinement of vague clusters",
)
@click.option(
    "--density-threshold",
    default=1.0,
    help="Density threshold for identifying vague clusters",
)
@click.option("--max-depth", default=2, help="Maximum recursion depth for refinement")
@click.option(
    "--model",
    default="clip",
    type=click.Choice(sorted(EMBEDDING_MODELS)),
    help="Embedding model to use if generating embeddings",
)
def analyze(
    image_dir,
    embeddings,
    sample_size,
    batch_size,
    output_dir,
    min_cluster_size,
    hierarchical,
    density_threshold,
    max_depth,
    model,
):
    """Analyze clustering results and generate statistics and visualizations."""
    import os
    from datetime import datetime

    from .analysis import AnalyzeOptions, run
    from .rr import init_rerun

    init_rerun(os.getenv("RERUN_ADDR"))

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"cluster_analysis_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    options = AnalyzeOptions(
        image_dir=image_dir,
        embeddings=embeddings,
        sample_size=sample_size,
        batch_size=batch_size,
        min_cluster_size=min_cluster_size,
        model=model,
    )
    run(options, output_path, hierarchical, density_threshold, max_depth)
