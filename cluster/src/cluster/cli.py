import glob
import logging
import os
import shutil
from pathlib import Path

# installed packages
import click
import numpy as np
from dotenv import load_dotenv
from PIL import Image

# local source
from common import ASSETS_DIR, PROJECT_ROOT, run_app
from embed import get_device

# TODO: put some default images in `./images/` as exampels
DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
CLUSTERS_PATH = "glashelder.clusters"
PACKAGE_ROOT = PROJECT_ROOT / "cluster"
EMBEDDINGS_PATH = PROJECT_ROOT / "embed" / "embeddings.pkl"

logger = logging.getLogger(__name__)


def resize_image_to_max_dimension(src_path: str, dst_path: str, max_dim: int = 512):
    """
    Resize an image to fit within max_dim x max_dim while preserving aspect ratio.

    Args:
        src_path: Source image path
        dst_path: Destination image path
        max_dim: Maximum width or height in pixels
    """
    img = Image.open(src_path)
    quality = 75

    # Calculate new dimensions while preserving aspect ratio
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        # Image is already small enough, just copy it
        img.save(dst_path, quality=quality)
        return

    # Calculate scaling factor
    scale = min(max_dim / width, max_dim / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize using high-quality resampling
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_img.save(dst_path, quality=quality)


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
    from collections import Counter
    from datetime import datetime

    from common.paths import rebase_paths
    from embed import Store, is_store, read_embeddings

    from .cluster import create_clusterer, reduce_dimensions, soft_memberships
    from .result import Clustering, fingerprint, save
    from .rr import init_rerun, log_clustered_points, log_images_3d, log_reduced

    init_rerun(os.getenv("RERUN_ADDR"))

    logger.info("Loading embeddings")
    data = read_embeddings(embeddings)
    refs: list[str] = data["images"]
    root = image_dir if image_dir is not None else data["root"]
    image_paths = rebase_paths(refs, root)
    edits = Store.open(embeddings).edits if is_store(embeddings) else None
    # UMAP needs all of it in memory, as float32
    vectors = data["embeddings"].float().numpy()
    logger.info(f"Loaded embeddings from model: {data.get('model', 'unknown')}")

    logger.info("Reducing dimensions")
    reduced_embeddings = reduce_dimensions(vectors, dimensions, n_neighbors, min_dist)
    visualization_embeddings = reduce_dimensions(
        reduced_embeddings, 3, n_neighbors, min_dist
    )

    if reduce:
        log_reduced(visualization_embeddings, count=20000)
        return

    flat_embeddings = reduce_dimensions(reduced_embeddings, 2, n_neighbors, min_dist)

    logger.info("cluster points")
    clusterer = create_clusterer(
        reduced_embeddings, min_cluster_size, min_samples, prediction_data=soft
    )
    labels = clusterer.labels_
    if soft:
        logger.info("Computing soft cluster memberships")
        soft_labels, soft_scores = soft_memberships(clusterer, soft_top)
    else:
        soft_labels = soft_scores = None
    sizes = Counter(int(label) for label in labels)
    noise = sizes.pop(-1, 0)

    save(
        output,
        Clustering(
            refs=refs,
            labels=labels,
            probabilities=clusterer.probabilities_,
            outlier_scores=clusterer.outlier_scores_,
            coords2d=flat_embeddings,
            coords3d=visualization_embeddings,
            reduced=reduced_embeddings,
            soft_labels=soft_labels,
            soft_scores=soft_scores,
            meta={
                "created": datetime.now().isoformat(timespec="seconds"),
                "embeddings": str(Path(embeddings).resolve()),
                "model": data["model"],
                "model_id": data["model_id"],
                "root": data["root"],
                "rows": len(refs),
                "edits": edits,
                "refs_hash": fingerprint(refs, edits),
                "umap": {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "dimensions": dimensions,
                },
                "hdbscan": {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                },
                "clusters": len(sizes),
                "noise": noise,
                "persistence": [float(p) for p in clusterer.cluster_persistence_],
                "soft": {"top": soft_top} if soft else None,
            },
        ),
    )
    logger.info(f"Saved the clustering to {output}")

    # log to rerun
    logger.info("logging clustered points to rerun")
    log_clustered_points(visualization_embeddings, labels, count=count)
    log_images_3d(visualization_embeddings, image_paths, 500)

    logger.info(f"Number of clusters: {len(sizes)}")
    logger.info(f"Number of noise points: {noise}")
    logger.info("Cluster sizes:")
    for label, size in sorted(sizes.items()):
        logger.info(f"  Cluster {label}: {size} points")

    if not output_files:
        return

    output_dir = Path(CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # For each cluster, copy up to 4 sample images
    for cluster_id in sorted(sizes):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        cluster_dir = output_dir / f"cluster_{cluster_id}"
        shutil.rmtree(cluster_dir, ignore_errors=True)
        cluster_dir.mkdir()

        sample_indices = cluster_indices[:4]
        for idx in sample_indices:
            src_path = image_paths[idx]
            dst_path = cluster_dir / Path(src_path).name
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        print(
            f"Cluster {cluster_id}: {len(cluster_indices)} total images, saved {len(sample_indices)} samples"
        )

    # Optionally handle noise points
    noise_indices = np.flatnonzero(labels == -1)
    if len(noise_indices):
        noise_dir = output_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        # Save up to 2 noise samples
        for idx in noise_indices[:2]:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = noise_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        print(
            f"Noise: {len(noise_indices)} total images, saved {min(2, len(noise_indices))} samples"
        )

    print(f"\nSamples saved to: {output_dir}")


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
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5001, help="Port to bind to")
@click.option(
    "--auth", is_flag=True, help="Enable secret url parameter based authentication."
)
@click.option("--debug", is_flag=True, help="Run Flask in debug mode")
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the site in a browser on startup",
)
def serve(
    clusters: Path | None,
    image_dir: str | None,
    host: str,
    port: int,
    auth: bool,
    debug: bool,
    open_browser: bool,
):
    """Show a clustering result in the browser, and give its clusters names and groups.

    Names and groups are saved in the result, in `annotations.json`.
    """
    from .web import create_app

    if clusters is None:
        raise click.UsageError("give --clusters, or set paths.clusters")
    app = create_app(clusters, image_dir, auth=auth)
    run_app(app, host, port, open_browser=open_browser, debug=debug)


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
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
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
    from datetime import datetime

    from embed import (
        CLIP_MODEL_ID,
        get_embedder,
        load_embeddings,
    )

    from .cluster import (
        calculate_cluster_statistics,
        create_2d_visualization,
        create_3d_visualization,
        create_clusterer,
        is_noise,
        noise_mask,
        reduce_dimensions,
        save_cluster_statistics,
    )
    from .rr import init_rerun, log_clustered_points_density

    init_rerun(os.getenv("RERUN_ADDR"))

    device = get_device()

    # Add timestamp to output directory if not explicitly specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"cluster_analysis_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load or generate embeddings
    if embeddings is not None and os.path.exists(embeddings):
        logger.info("Loading embeddings")
        data = load_embeddings(embeddings, root=image_dir)
        image_paths = data["images"][:sample_size]
        embeddings_tensor = data["embeddings"][:sample_size].float().to(device)
        embedding_model_name = data.get("model", "clip")
        embedding_model_id = data.get("model_id", CLIP_MODEL_ID)
        logger.info(f"Loaded embeddings from model: {embedding_model_name}")
    else:
        logger.info(f"Getting embedding model: {model}")
        embedder = get_embedder(model)
        embedder.load(device)
        embedding_model_name = embedder.name
        embedding_model_id = embedder.model_id
        input_paths = list(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))[
            :sample_size
        ]
        logger.info("Generating image embeddings...")
        embeddings_tensor, image_paths = embedder.encode_images(input_paths, batch_size)
        if len(image_paths) < len(input_paths):
            logger.warning(
                f"Skipped {len(input_paths) - len(image_paths)} images due to load errors"
            )

    logger.info(f"Processing {len(image_paths)} images")

    # Reduce dimensions for clustering (30D) and visualization (2D and 3D)
    logger.info("Reducing dimensions for clustering (30D)...")
    reduced_embeddings_30d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 30)

    logger.info("Reducing dimensions for 2D visualization...")
    reduced_embeddings_2d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 2)

    logger.info("Reducing dimensions for 3D visualization...")
    reduced_embeddings_3d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 3)

    # Perform clustering
    logger.info(f"Clustering with min_cluster_size={min_cluster_size}...")
    clusterer = create_clusterer(
        reduced_embeddings_30d, min_cluster_size=min_cluster_size
    )
    labels = clusterer.labels_

    # Apply hierarchical refinement if enabled
    if hierarchical:
        from .cluster import HierarchicalConfig, hierarchical_cluster

        config = HierarchicalConfig(
            enabled=True,
            density_threshold=density_threshold,
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Starting hierarchical refinement...")
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  Density threshold: {density_threshold}")
        logger.info(f"  Max depth: {max_depth}")
        logger.info(f"  Min cluster size: {min_cluster_size}")

        # refined clusters come with coordinates from their own UMAP fit; keep the
        # global 30D fit so every cluster's statistics are measured in one space
        labels, _, refinement_tree = hierarchical_cluster(
            embeddings_tensor.cpu().numpy(), labels, reduced_embeddings_30d, config
        )

    # Calculate statistics
    logger.info("Calculating cluster statistics...")
    stats = calculate_cluster_statistics(reduced_embeddings_30d, labels, clusterer)

    # log to rerun
    log_clustered_points_density(
        reduced_embeddings_3d, labels, stats, 30000, "clusters"
    )

    # Add hierarchical metadata if used
    if hierarchical:
        stats["refinement_tree"] = refinement_tree
        stats["metadata_hierarchical"] = {
            "enabled": True,
            "density_threshold": density_threshold,
            "max_depth": max_depth,
        }

    # Define output paths
    json_path = output_path / "cluster_stats.json"
    plot_2d_path = output_path / "clusters_2d.png"
    plot_3d_path = output_path / "clusters_3d.png"

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    create_2d_visualization(reduced_embeddings_2d, labels, str(plot_2d_path))
    create_3d_visualization(reduced_embeddings_3d, labels, str(plot_3d_path))

    # Save sample images for each cluster
    logger.info("Saving sample images for each cluster...")
    samples_dir = output_path / "cluster_samples"
    samples_dir.mkdir(exist_ok=True)

    # Convert labels to strings for consistent comparison
    labels_str = [str(label) for label in labels]

    # Get unique clusters (excluding noise and .noise suffix)
    unique_clusters = sorted(
        [label for label in set(labels_str) if not is_noise(label)],
        key=lambda x: (x.count("."), x),
    )  # Sort by depth, then name

    # For each cluster, copy sample images
    for cluster_id in unique_clusters:
        # Create cluster directory (handle hierarchical labels with dots)
        cluster_dir = samples_dir / f"cluster_{cluster_id}"
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(exist_ok=True, parents=True)

        # Get indices of images in this cluster
        cluster_indices = [
            i for i, label in enumerate(labels_str) if label == cluster_id
        ]

        # Select up to 4 samples
        sample_indices = cluster_indices[:4]

        # Copy and resize images
        for idx in sample_indices:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = cluster_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        logger.info(
            f"Cluster {cluster_id}: {len(cluster_indices)} total images, saved {len(sample_indices)} samples"
        )

    # Handle noise points (both "-1" and ".noise" suffix)
    noise_indices = np.flatnonzero(noise_mask(labels_str)).tolist()
    if noise_indices:
        noise_dir = samples_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        # Save up to 2 noise samples
        for idx in noise_indices[:2]:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = noise_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        logger.info(
            f"Noise: {len(noise_indices)} total images, saved {min(2, len(noise_indices))} samples"
        )

    # Also save hierarchical .noise clusters separately
    if hierarchical:
        noise_clusters = sorted(
            [label for label in set(labels_str) if label.endswith(".noise")]
        )
        for noise_cluster_id in noise_clusters:
            noise_cluster_dir = samples_dir / f"cluster_{noise_cluster_id}"
            noise_cluster_dir.mkdir(exist_ok=True, parents=True)

            noise_cluster_indices = [
                i for i, label in enumerate(labels_str) if label == noise_cluster_id
            ]
            for idx in noise_cluster_indices[:4]:
                src_path = image_paths[idx]
                filename = Path(src_path).name
                dst_path = noise_cluster_dir / filename
                resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

            logger.info(
                f"Cluster {noise_cluster_id}: {len(noise_cluster_indices)} total images, saved {min(4, len(noise_cluster_indices))} samples"
            )

    # Add metadata to stats (including visualization paths)
    stats["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(image_paths),
        "min_cluster_size": min_cluster_size,
        "embedding_model": embedding_model_name,
        "embedding_model_id": embedding_model_id,
        "reduction_dims": 30,
        "output_directory": str(output_path.resolve()),
        "visualizations": {
            "plot_2d": str(plot_2d_path.resolve()),
            "plot_3d": str(plot_3d_path.resolve()),
        },
        "sample_images": str(samples_dir.resolve()),
    }

    # Print hierarchical refinement summary if applicable
    if hierarchical and refinement_tree:
        logger.info("\n" + "=" * 60)
        logger.info("HIERARCHICAL CLUSTERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Refinement depth: {max_depth}")
        logger.info(f"Clusters refined: {len(refinement_tree)}")

        total_sub_clusters = 0
        for parent_id, info in sorted(
            refinement_tree.items(), key=lambda x: (x[0].count("."), x[0])
        ):
            sub_cluster_count = len(
                [c for c in info["refined_into"] if not c.endswith(".noise")]
            )
            total_sub_clusters += sub_cluster_count

            logger.info(f"\nCluster {parent_id} (depth {info['depth']}):")
            logger.info(
                f"  Original: {info['original_size']} pts, density {info['original_density']:.2f}"
            )
            logger.info(f"  Refined into {len(info['refined_into'])} sub-clusters:")

            # Show breakdown of sub-clusters with their stats
            for sub_label in sorted(
                info["refined_into"], key=lambda x: (x.count("."), x)
            ):
                # Find stats for this sub-cluster
                sub_stats = next(
                    (c for c in stats["clusters"] if str(c["cluster_id"]) == sub_label),
                    None,
                )
                if sub_stats:
                    status = (
                        "✓"
                        if sub_stats["density"] >= density_threshold
                        else "→ needs refinement"
                    )
                    logger.info(
                        f"    ├─ {sub_label}: {sub_stats['size']} pts "
                        f"(density {sub_stats['density']:.2f}) {status}"
                    )
                else:
                    # It's a noise cluster
                    noise_count = len(
                        [i for i, label in enumerate(labels_str) if label == sub_label]
                    )
                    logger.info(f"    └─ {sub_label}: {noise_count} pts")

        logger.info(f"\nTotal new sub-clusters created: {total_sub_clusters}")

        # Calculate density improvements
        density_improvements = []
        for _parent_id, info in refinement_tree.items():
            original_density = info["original_density"]
            for sub_label in info["refined_into"]:
                if not sub_label.endswith(".noise"):
                    sub_stats = next(
                        (
                            c
                            for c in stats["clusters"]
                            if str(c["cluster_id"]) == sub_label
                        ),
                        None,
                    )
                    if sub_stats and original_density > 0:
                        improvement = sub_stats["density"] / original_density
                        density_improvements.append(improvement)

        if density_improvements:
            avg_improvement = sum(density_improvements) / len(density_improvements)
            logger.info(f"Average density improvement: {avg_improvement:.2f}x")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['overall']['n_samples']}")
    logger.info(f"Number of clusters: {stats['overall']['n_clusters']}")
    logger.info(
        f"Noise points: {stats['overall']['n_noise']} ({stats['overall']['noise_ratio'] * 100:.2f}%)"
    )
    if stats["overall"]["silhouette_score"] is not None:
        logger.info(f"Silhouette score: {stats['overall']['silhouette_score']:.4f}")

    logger.info("\nTop 10 clusters by size:")
    for i, cluster in enumerate(stats["clusters"][:10], 1):
        logger.info(
            f"  {i}. Cluster {cluster['cluster_id']}: "
            f"{cluster['size']} points ({cluster['percentage']:.2f}%), "
            f"compactness={cluster['compactness']:.4f}, "
            f"density={cluster['density']:.4f}"
        )

    # Save statistics to JSON (after adding visualization paths)
    save_cluster_statistics(stats, json_path)

    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis complete! Results saved to: {output_path}")
    logger.info(f"  - Statistics: {json_path}")
    logger.info(f"  - 2D plot: {output_path / 'clusters_2d.png'}")
    logger.info(f"  - 3D plot: {output_path / 'clusters_3d.png'}")
    logger.info(f"  - Sample images: {samples_dir}")
    logger.info("=" * 60)
