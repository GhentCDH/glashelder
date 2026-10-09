import logging
from dataclasses import dataclass

import numpy as np

from .reduce import create_clusterer, reduce_dimensions
from .stats import calculate_cluster_statistics

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical clustering."""

    density_threshold: float = 1.0
    max_depth: int = 2
    min_cluster_size: int = 50


def should_recluster(cluster_stats: dict, config: HierarchicalConfig) -> bool:
    """
    Determine if a cluster needs refinement.

    Returns True if cluster is both large and vague.
    """
    return cluster_stats["density"] < config.density_threshold


def cluster_indices_for(labels: np.ndarray, cluster_id: str | int) -> np.ndarray:
    """
    Indices of the points belonging to `cluster_id`.

    Handles both numeric (0, 1, 2) and hierarchical ("2.0", "2.1") labels.
    """
    cluster_id_str = str(cluster_id)
    mask = np.array([str(label) == cluster_id_str for label in labels])
    indices = np.where(mask)[0]
    logger.debug(f"Extracted {len(indices)} points for cluster '{cluster_id_str}'")
    return indices


def refine_cluster(
    original_embeddings: np.ndarray,
    cluster_indices: np.ndarray,
    parent_label: str,
    config: HierarchicalConfig,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Sub-cluster a problematic cluster.

    Steps:
    1. Extract embeddings for cluster members
    2. Run UMAP (30D) on subset
    3. Run HDBSCAN with adaptive min_cluster_size
    4. Create hierarchical labels
    5. Handle failure cases (no clusters, only noise)

    Returns: (sub_labels, sub_reduced_embeddings) or (None, None) if refinement fails
    """
    # Extract subset embeddings
    subset_embeddings = original_embeddings[cluster_indices]

    # Re-run UMAP on subset
    logger.info(f"  Running UMAP on {len(cluster_indices)} points...")
    subset_reduced = reduce_dimensions(subset_embeddings, dimensions=30)

    # Adaptive min_cluster_size
    adaptive_min_size = max(config.min_cluster_size, int(len(cluster_indices) * 0.05))
    logger.info(f"  Running HDBSCAN with min_cluster_size={adaptive_min_size}...")

    # Re-run HDBSCAN
    clusterer = create_clusterer(subset_reduced, min_cluster_size=adaptive_min_size)
    sub_labels = clusterer.labels_

    # Check for valid splits
    n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    if n_clusters <= 1:
        # No meaningful split found
        logger.info(f"  No meaningful split found (got {n_clusters} clusters)")
        return None, None

    # Create hierarchical labels
    hierarchical_labels = []
    for label in sub_labels:
        if label == -1:
            new_label = f"{parent_label}.noise"
        else:
            new_label = f"{parent_label}.{label}"
        hierarchical_labels.append(new_label)

    logger.info(f"  Successfully split into {n_clusters} sub-clusters")
    logger.debug(
        f"  Parent label: '{parent_label}', sample sub-labels: {sorted(set(hierarchical_labels))[:5]}"
    )
    return np.array(hierarchical_labels), subset_reduced


def hierarchical_cluster(
    original_embeddings: np.ndarray,
    initial_labels: np.ndarray,
    initial_reduced_embeddings: np.ndarray,
    config: HierarchicalConfig,
    depth: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Recursively refine clusters that meet criteria.

    Returns: (hierarchical_labels, hierarchical_reduced_embeddings, refinement_tree)
    """
    # Calculate statistics for current clustering
    stats = calculate_cluster_statistics(
        initial_reduced_embeddings, initial_labels, None
    )

    # Initialize refinement tree
    refinement_tree = {}

    # Base case: max depth reached
    if depth >= config.max_depth:
        logger.info(f"Max depth {config.max_depth} reached, stopping refinement")
        return initial_labels, initial_reduced_embeddings, refinement_tree

    # Convert numeric labels to strings for consistency
    # Use object dtype to avoid string truncation issues with hierarchical labels
    current_labels = np.array([str(label) for label in initial_labels], dtype=object)
    current_reduced = initial_reduced_embeddings.copy()

    # Identify clusters to refine
    clusters_to_refine = []
    for cluster_stat in stats["clusters"]:
        if should_recluster(cluster_stat, config):
            clusters_to_refine.append(cluster_stat["cluster_id"])

    if not clusters_to_refine:
        # No clusters need refinement
        logger.info(f"No clusters meet refinement criteria at depth {depth}")
        return current_labels, current_reduced, refinement_tree

    logger.info(f"\n{'=' * 60}")
    logger.info(f"REFINEMENT ITERATION {depth + 1}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Clusters to refine: {len(clusters_to_refine)}")

    # Refine each problematic cluster
    for cluster_id in clusters_to_refine:
        cluster_id_str = str(cluster_id)

        # Get original stats for this cluster
        original_stats = next(
            c for c in stats["clusters"] if c["cluster_id"] == cluster_id
        )
        logger.info(
            f"\nCluster {cluster_id_str} ({original_stats['size']} pts, density {original_stats['density']:.2f}):"
        )

        # Extract cluster indices
        cluster_indices = cluster_indices_for(current_labels, cluster_id_str)
        logger.debug(f"  Extracted {len(cluster_indices)} points for refinement")

        # Attempt refinement
        sub_labels, sub_reduced = refine_cluster(
            original_embeddings, cluster_indices, cluster_id_str, config
        )

        if sub_labels is None or sub_reduced is None:
            # Refinement failed, keep original cluster
            logger.info(f"  Keeping cluster {cluster_id_str} as-is")
            continue

        # Update labels for refined cluster
        for i, idx in enumerate(cluster_indices):
            current_labels[idx] = sub_labels[i]
            current_reduced[idx] = sub_reduced[i]

        # Validate: ensure the original cluster label no longer exists
        remaining_original = np.sum(current_labels == cluster_id_str)
        if remaining_original > 0:
            logger.warning(
                f"  WARNING: {remaining_original} points still have original label {cluster_id_str}!"
            )

        # Record refinement
        unique_sub_labels = sorted(set(sub_labels))
        refinement_tree[cluster_id_str] = {
            "original_size": int(original_stats["size"]),
            "original_density": float(original_stats["density"]),
            "refined_into": [str(label) for label in unique_sub_labels],
            "depth": depth,
        }

        # Log sub-cluster breakdown
        for sub_label in unique_sub_labels:
            count = int(np.sum(sub_labels == sub_label))
            logger.info(f"  └─ {sub_label}: {count} pts")
            # Validate label format
            if sub_label != f"{cluster_id_str}.noise" and not sub_label.startswith(
                f"{cluster_id_str}."
            ):
                logger.error(
                    f"  ERROR: Malformed sub-label '{sub_label}' for parent '{cluster_id_str}'"
                )

    # Recurse on new sub-clusters
    if depth + 1 < config.max_depth:
        logger.info("\nChecking sub-clusters for further refinement...")
        current_labels, current_reduced, sub_tree = hierarchical_cluster(
            original_embeddings, current_labels, current_reduced, config, depth + 1
        )

        # Merge refinement trees
        for key, value in sub_tree.items():
            if "." in key:  # It's a sub-cluster
                parent = key.rsplit(".", 1)[0]
                if parent in refinement_tree:
                    if "sub_clusters" not in refinement_tree[parent]:
                        refinement_tree[parent]["sub_clusters"] = {}
                    refinement_tree[parent]["sub_clusters"][key] = value
            else:
                refinement_tree[key] = value

    # Log final label distribution at this depth
    if depth == 0:
        unique_final_labels = sorted(
            set(current_labels), key=lambda x: (x.count("."), x)
        )
        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL LABEL DISTRIBUTION")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total unique labels: {len(unique_final_labels)}")
        hierarchical_labels = [
            label for label in unique_final_labels if "." in label and label != "-1"
        ]
        logger.info(f"Hierarchical labels: {len(hierarchical_labels)}")
        logger.info(f"Sample hierarchical labels: {hierarchical_labels[:10]}")

    return current_labels, current_reduced, refinement_tree
