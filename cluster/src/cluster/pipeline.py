"""The `cluster cluster` command's pipeline: load embeddings, reduce, cluster, save."""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from common.paths import rebase_paths
from embed import Store, is_store, read_embeddings

from .reduce import create_clusterer, reduce_dimensions, soft_memberships
from .result import Clustering, fingerprint, save
from .rr import log_clustered_points, log_images_3d, log_reduced
from .samples import export_samples

logger = logging.getLogger(__name__)


@dataclass
class ClusterOptions:
    embeddings: str | Path
    image_dir: str | None
    n_neighbors: int
    min_dist: float
    dimensions: int
    min_cluster_size: int
    min_samples: int | None
    soft: bool
    soft_top: int


def run(options: ClusterOptions) -> tuple[Clustering, list[str]]:
    """Load embeddings, reduce dimensions and cluster them.

    Returns the `Clustering` and the image paths of its rows, rebased to
    `options.image_dir`.
    """
    logger.info("Loading embeddings")
    data = read_embeddings(options.embeddings)
    refs: list[str] = data["images"]
    root = options.image_dir if options.image_dir is not None else data["root"]
    image_paths = rebase_paths(refs, root)
    edits = (
        Store.open(options.embeddings).edits if is_store(options.embeddings) else None
    )
    # UMAP needs all of it in memory, as float32
    vectors = data["embeddings"].float().numpy()
    logger.info(f"Loaded embeddings from model: {data.get('model', 'unknown')}")

    logger.info("Reducing dimensions")
    reduced_embeddings = reduce_dimensions(
        vectors, options.dimensions, options.n_neighbors, options.min_dist
    )
    visualization_embeddings = reduce_dimensions(
        reduced_embeddings, 3, options.n_neighbors, options.min_dist
    )
    flat_embeddings = reduce_dimensions(
        reduced_embeddings, 2, options.n_neighbors, options.min_dist
    )

    logger.info("cluster points")
    clusterer = create_clusterer(
        reduced_embeddings,
        options.min_cluster_size,
        options.min_samples,
        prediction_data=options.soft,
    )
    labels = clusterer.labels_
    if options.soft:
        logger.info("Computing soft cluster memberships")
        soft_labels, soft_scores = soft_memberships(clusterer, options.soft_top)
    else:
        soft_labels = soft_scores = None

    sizes = Counter(int(label) for label in labels)
    noise = sizes.pop(-1, 0)

    clustering = Clustering(
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
            "embeddings": str(Path(options.embeddings).resolve()),
            "model": data["model"],
            "model_id": data["model_id"],
            "root": data["root"],
            "rows": len(refs),
            "edits": edits,
            "refs_hash": fingerprint(refs, edits),
            "umap": {
                "n_neighbors": options.n_neighbors,
                "min_dist": options.min_dist,
                "dimensions": options.dimensions,
            },
            "hdbscan": {
                "min_cluster_size": options.min_cluster_size,
                "min_samples": options.min_samples,
            },
            "clusters": len(sizes),
            "noise": noise,
            "persistence": [float(p) for p in clusterer.cluster_persistence_],
            "soft": {"top": options.soft_top} if options.soft else None,
        },
    )
    return clustering, image_paths


def reduce_only(options: ClusterOptions, count: int) -> None:
    """Reduce dimensions and log the 3D result to rerun, without clustering."""
    data = read_embeddings(options.embeddings)
    vectors = data["embeddings"].float().numpy()
    reduced_embeddings = reduce_dimensions(
        vectors, options.dimensions, options.n_neighbors, options.min_dist
    )
    visualization_embeddings = reduce_dimensions(
        reduced_embeddings, 3, options.n_neighbors, options.min_dist
    )
    log_reduced(visualization_embeddings, count=count)


def save_clustering(output: Path, clustering: Clustering) -> None:
    save(output, clustering)
    logger.info(f"Saved the clustering to {output}")


def log_cluster_sizes(clustering: Clustering) -> None:
    sizes = Counter(int(label) for label in clustering.labels)
    noise = sizes.pop(-1, 0)
    logger.info(f"Number of clusters: {len(sizes)}")
    logger.info(f"Number of noise points: {noise}")
    logger.info("Cluster sizes:")
    for label, size in sorted(sizes.items()):
        logger.info(f"  Cluster {label}: {size} points")


def log_to_rerun(clustering: Clustering, image_paths: list[str], count: int) -> None:
    logger.info("logging clustered points to rerun")
    log_clustered_points(clustering.coords3d, clustering.labels, count=count)
    log_images_3d(clustering.coords3d, image_paths, 500)


def export_cluster_samples(
    clustering: Clustering, image_paths: list[str], output_dir: Path
) -> None:
    """Save up to 4 sample images per cluster, and up to 2 for noise, under `output_dir`."""
    labels = clustering.labels
    sizes = Counter(int(label) for label in labels)
    sizes.pop(-1, None)
    output_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(sizes):
        indices = np.flatnonzero(labels == cluster_id)
        export_samples(
            cluster_id, indices, image_paths, output_dir / f"cluster_{cluster_id}", 4
        )

    noise_indices = np.flatnonzero(labels == -1)
    if len(noise_indices):
        export_samples(
            "noise", noise_indices, image_paths, output_dir / "noise", 2, clean=False
        )

    print(f"\nSamples saved to: {output_dir}")
