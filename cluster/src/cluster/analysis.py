"""The `cluster analyze` command: stats, plots and samples for an ad hoc clustering run."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from common.images import list_images
from embed import CLIP_MODEL_ID, get_device, get_embedder, load_embeddings

from .hierarchical import HierarchicalConfig, hierarchical_cluster
from .plots import create_2d_visualization, create_3d_visualization
from .reduce import create_clusterer, reduce_dimensions
from .rr import log_clustered_points_density
from .samples import export_samples
from .stats import (
    calculate_cluster_statistics,
    is_noise,
    noise_mask,
    save_cluster_statistics,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeOptions:
    image_dir: str
    embeddings: str | None
    sample_size: int
    batch_size: int
    min_cluster_size: int
    model: str


def load_or_generate(options: AnalyzeOptions):
    """Load embeddings from a store, or generate them from `image_dir`.

    Returns `(embeddings_tensor, image_paths, model_name, model_id)`.
    """
    device = get_device()

    if options.embeddings is not None and os.path.exists(options.embeddings):
        logger.info("Loading embeddings")
        data = load_embeddings(options.embeddings, root=options.image_dir)
        image_paths = data["images"][: options.sample_size]
        embeddings_tensor = data["embeddings"][: options.sample_size].float().to(device)
        model_name = data.get("model", "clip")
        model_id = data.get("model_id", CLIP_MODEL_ID)
        logger.info(f"Loaded embeddings from model: {model_name}")
        return embeddings_tensor, image_paths, model_name, model_id

    logger.info(f"Getting embedding model: {options.model}")
    embedder = get_embedder(options.model)
    embedder.load(device)
    input_paths = [
        str(p)
        for p in list_images(options.image_dir, suffixes=(".jpg",))[
            : options.sample_size
        ]
    ]
    logger.info("Generating image embeddings...")
    embeddings_tensor, image_paths = embedder.encode_images(
        input_paths, options.batch_size
    )
    if len(image_paths) < len(input_paths):
        logger.warning(
            f"Skipped {len(input_paths) - len(image_paths)} images due to load errors"
        )
    return embeddings_tensor, image_paths, embedder.name, embedder.model_id


def reduce_and_cluster(embeddings_tensor, min_cluster_size: int):
    """Reduce for clustering (30D) and for the 2D/3D plots, and cluster the 30D space."""
    vectors = embeddings_tensor.cpu().numpy()

    logger.info("Reducing dimensions for clustering (30D)...")
    reduced_30d = reduce_dimensions(vectors, 30)
    logger.info("Reducing dimensions for 2D visualization...")
    reduced_2d = reduce_dimensions(vectors, 2)
    logger.info("Reducing dimensions for 3D visualization...")
    reduced_3d = reduce_dimensions(vectors, 3)

    logger.info(f"Clustering with min_cluster_size={min_cluster_size}...")
    clusterer = create_clusterer(reduced_30d, min_cluster_size=min_cluster_size)
    return clusterer, reduced_30d, reduced_2d, reduced_3d


def refine(
    embeddings_tensor,
    labels,
    reduced_30d,
    density_threshold: float,
    max_depth: int,
    min_cluster_size: int,
):
    config = HierarchicalConfig(
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
        embeddings_tensor.cpu().numpy(), labels, reduced_30d, config
    )
    return labels, refinement_tree


def save_samples(labels, image_paths, output_path: Path, hierarchical: bool) -> Path:
    """Export sample images per cluster, noise, and (if hierarchical) `.noise` sub-clusters."""
    samples_dir = output_path / "cluster_samples"
    samples_dir.mkdir(exist_ok=True)

    labels_str = [str(label) for label in labels]
    logger.info("Saving sample images for each cluster...")

    unique_clusters = sorted(
        (label for label in set(labels_str) if not is_noise(label)),
        key=lambda x: (x.count("."), x),
    )
    for cluster_id in unique_clusters:
        indices = [i for i, label in enumerate(labels_str) if label == cluster_id]
        export_samples(
            cluster_id, indices, image_paths, samples_dir / f"cluster_{cluster_id}", 4
        )

    noise_indices = np.flatnonzero(noise_mask(labels_str)).tolist()
    if noise_indices:
        export_samples(
            "noise", noise_indices, image_paths, samples_dir / "noise", 2, clean=False
        )

    if hierarchical:
        noise_clusters = sorted(
            label for label in set(labels_str) if label.endswith(".noise")
        )
        for noise_cluster_id in noise_clusters:
            indices = [
                i for i, label in enumerate(labels_str) if label == noise_cluster_id
            ]
            export_samples(
                noise_cluster_id,
                indices,
                image_paths,
                samples_dir / f"cluster_{noise_cluster_id}",
                4,
            )

    return samples_dir


def log_refinement_summary(
    refinement_tree: dict, stats: dict, labels_str: list[str], density_threshold: float
) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("HIERARCHICAL CLUSTERING SUMMARY")
    logger.info("=" * 60)
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

        for sub_label in sorted(info["refined_into"], key=lambda x: (x.count("."), x)):
            sub_stats = next(
                (c for c in stats["clusters"] if str(c["cluster_id"]) == sub_label),
                None,
            )
            if sub_stats:
                status = (
                    "done"
                    if sub_stats["density"] >= density_threshold
                    else "needs refinement"
                )
                logger.info(
                    f"    - {sub_label}: {sub_stats['size']} pts "
                    f"(density {sub_stats['density']:.2f}) {status}"
                )
            else:
                noise_count = len(
                    [i for i, label in enumerate(labels_str) if label == sub_label]
                )
                logger.info(f"    - {sub_label}: {noise_count} pts")

    logger.info(f"\nTotal new sub-clusters created: {total_sub_clusters}")

    density_improvements = []
    for info in refinement_tree.values():
        original_density = info["original_density"]
        for sub_label in info["refined_into"]:
            if sub_label.endswith(".noise"):
                continue
            sub_stats = next(
                (c for c in stats["clusters"] if str(c["cluster_id"]) == sub_label),
                None,
            )
            if sub_stats and original_density > 0:
                density_improvements.append(sub_stats["density"] / original_density)

    if density_improvements:
        avg_improvement = sum(density_improvements) / len(density_improvements)
        logger.info(f"Average density improvement: {avg_improvement:.2f}x")


def log_summary(stats: dict) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['overall']['n_samples']}")
    logger.info(f"Number of clusters: {stats['overall']['n_clusters']}")
    logger.info(
        f"Noise points: {stats['overall']['n_noise']} "
        f"({stats['overall']['noise_ratio'] * 100:.2f}%)"
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


def run(
    options: AnalyzeOptions,
    output_path: Path,
    hierarchical: bool,
    density_threshold: float,
    max_depth: int,
) -> dict:
    """Run the full analyze pipeline and return its statistics dict."""
    embeddings_tensor, image_paths, model_name, model_id = load_or_generate(options)
    logger.info(f"Processing {len(image_paths)} images")

    clusterer, reduced_30d, reduced_2d, reduced_3d = reduce_and_cluster(
        embeddings_tensor, options.min_cluster_size
    )
    labels = clusterer.labels_

    refinement_tree = None
    if hierarchical:
        labels, refinement_tree = refine(
            embeddings_tensor,
            labels,
            reduced_30d,
            density_threshold,
            max_depth,
            options.min_cluster_size,
        )

    logger.info("Calculating cluster statistics...")
    stats = calculate_cluster_statistics(reduced_30d, labels, clusterer)

    log_clustered_points_density(reduced_3d, labels, stats, 30000, "clusters")

    if hierarchical:
        stats["refinement_tree"] = refinement_tree
        stats["metadata_hierarchical"] = {
            "enabled": True,
            "density_threshold": density_threshold,
            "max_depth": max_depth,
        }

    plot_2d_path = output_path / "clusters_2d.png"
    plot_3d_path = output_path / "clusters_3d.png"
    logger.info("\nGenerating visualizations...")
    create_2d_visualization(reduced_2d, labels, str(plot_2d_path))
    create_3d_visualization(reduced_3d, labels, str(plot_3d_path))

    samples_dir = save_samples(labels, image_paths, output_path, hierarchical)

    stats["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(image_paths),
        "min_cluster_size": options.min_cluster_size,
        "embedding_model": model_name,
        "embedding_model_id": model_id,
        "reduction_dims": 30,
        "output_directory": str(output_path.resolve()),
        "visualizations": {
            "plot_2d": str(plot_2d_path.resolve()),
            "plot_3d": str(plot_3d_path.resolve()),
        },
        "sample_images": str(samples_dir.resolve()),
    }

    if hierarchical and refinement_tree:
        labels_str = [str(label) for label in labels]
        log_refinement_summary(refinement_tree, stats, labels_str, density_threshold)

    log_summary(stats)

    json_path = output_path / "cluster_stats.json"
    save_cluster_statistics(stats, json_path)

    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis complete! Results saved to: {output_path}")
    logger.info(f"  - Statistics: {json_path}")
    logger.info(f"  - 2D plot: {plot_2d_path}")
    logger.info(f"  - 3D plot: {plot_3d_path}")
    logger.info(f"  - Sample images: {samples_dir}")
    logger.info("=" * 60)

    return stats
