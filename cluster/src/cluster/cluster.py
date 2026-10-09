import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN

# installed packages
from umap import UMAP

# local source

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical clustering."""

    enabled: bool = False
    density_threshold: float = 1.0
    max_depth: int = 2
    min_cluster_size: int = 50


def is_noise(label) -> bool:
    """Whether a cluster label marks noise.

    HDBSCAN marks noise with -1. Hierarchical refinement turns labels into strings,
    where noise is "-1" or "<parent>.noise".
    """
    text = str(label)
    return text == "-1" or text.endswith(".noise")


def noise_mask(labels) -> np.ndarray:
    """A boolean mask of the noise points in `labels`, numeric or hierarchical."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels == -1
    return np.array([is_noise(label) for label in labels], dtype=bool)


def reduce_dimensions(
    embeddings, dimensions: int = 30, n_neighbors: int = 30, min_dist: float = 0.0
):
    umap_model = UMAP(
        n_components=dimensions,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
    )
    return umap_model.fit_transform(embeddings)


def create_clusterer(
    latent,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    prediction_data: bool = False,
):
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=prediction_data,
    )
    clusterer.fit(latent)

    return clusterer


def soft_memberships(clusterer, top: int) -> tuple[np.ndarray, np.ndarray]:
    """The `top` strongest clusters of every point, with their membership scores.

    Returns `(labels, scores)`, both `(n, top)`, strongest first. The full membership
    matrix (points x clusters, float64) can be large, so only this much is kept.
    """
    from hdbscan.prediction import all_points_membership_vectors

    memberships = all_points_membership_vectors(clusterer)
    if memberships.ndim == 1:  # no clusters at all
        n = len(memberships)
        return np.full((n, top), -1, np.int32), np.zeros((n, top), np.float32)

    # column j is cluster label j; with fewer clusters than `top`, pad with -1
    k = min(top, memberships.shape[1])
    strongest = np.argpartition(-memberships, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(memberships, strongest, axis=1)
    order = np.argsort(-scores, axis=1)
    labels = np.full((len(memberships), top), -1, np.int32)
    values = np.zeros((len(memberships), top), np.float32)
    labels[:, :k] = np.take_along_axis(strongest, order, axis=1)
    values[:, :k] = np.take_along_axis(scores, order, axis=1)
    return labels, values


def create_scatter_plot(latent, labels, save_path: str = "clustering_results.png"):
    # Create a scatter plot
    plt.figure(figsize=(12, 8))

    # Use first 2 components for visualization
    x = latent[:, 0]
    y = latent[:, 1]

    # Create scatter plot with different colors for each cluster
    scatter = plt.scatter(x, y, c=labels, cmap="tab20", alpha=0.6, s=50)

    # Add colorbar
    plt.colorbar(scatter, label="Cluster")

    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title(
        f"HDBSCAN Clustering Results\n({len(set(labels)) - (1 if -1 in labels else 0)} clusters, {sum(labels == -1)} noise points)"
    )

    # Add grid
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save instead of show
    plt.close()  # Close the figure to free memory


def calculate_cluster_statistics(embeddings, labels, clusterer):
    """
    Calculate comprehensive statistics for clustering results.

    Args:
        embeddings: The reduced embeddings used for clustering
        labels: Cluster labels from HDBSCAN (can be numeric or string)
        clusterer: The fitted HDBSCAN clusterer object

    Returns:
        Dictionary containing overall and per-cluster statistics
    """
    # Convert all labels to strings for consistent handling
    # Use object dtype to avoid string truncation issues
    labels_str = np.array([str(label) for label in labels], dtype=object)
    unique_labels = sorted(
        set(labels_str), key=lambda x: (x.count("."), x)
    )  # Sort by depth, then name

    noise = noise_mask(labels_str)
    n_noise = int(noise.sum())
    n_clusters = sum(not is_noise(label) for label in unique_labels)
    n_samples = len(labels)

    # Overall statistics (ensure all values are native Python types for JSON serialization)
    overall_stats = {
        "n_clusters": int(n_clusters),
        "n_samples": int(n_samples),
        "n_noise": int(n_noise),
        "noise_ratio": float(n_noise / n_samples) if n_samples > 0 else 0.0,
    }

    # Calculate silhouette score if we have more than 1 cluster and less than all noise
    if n_clusters > 1 and n_noise < n_samples:
        try:
            from sklearn.metrics import silhouette_score

            # Only calculate silhouette for non-noise points
            non_noise_mask = ~noise
            if sum(non_noise_mask) > 0:
                # Create numeric labels for silhouette calculation
                non_noise_labels = labels_str[non_noise_mask]
                unique_non_noise = sorted(set(non_noise_labels))
                label_to_int = {label: i for i, label in enumerate(unique_non_noise)}
                numeric_labels = np.array(
                    [label_to_int[label] for label in non_noise_labels]
                )

                silhouette = silhouette_score(
                    embeddings[non_noise_mask], numeric_labels
                )
                overall_stats["silhouette_score"] = float(silhouette)
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            overall_stats["silhouette_score"] = None
    else:
        overall_stats["silhouette_score"] = None

    # Per-cluster statistics
    cluster_stats = []

    for label in unique_labels:
        if is_noise(label):
            continue  # Skip noise in per-cluster stats

        cluster_mask = labels_str == label
        cluster_points = embeddings[cluster_mask]
        cluster_size = int(sum(cluster_mask))

        # Calculate centroid
        centroid = np.mean(cluster_points, axis=0)

        # Calculate compactness (average distance to centroid)
        distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        compactness = float(np.mean(distances_to_centroid))
        std_distance = float(np.std(distances_to_centroid))
        max_distance = float(np.max(distances_to_centroid))

        # Calculate density (inverse of average pairwise distance)
        # For large clusters, sample to avoid memory issues
        if cluster_size > 1000:
            sample_idx = np.random.choice(cluster_size, 1000, replace=False)
            sample_points = cluster_points[sample_idx]
        else:
            sample_points = cluster_points

        if len(sample_points) > 1:
            from scipy.spatial.distance import pdist

            pairwise_distances = pdist(sample_points)
            avg_pairwise_distance = float(np.mean(pairwise_distances))
            density = (
                float(1.0 / avg_pairwise_distance) if avg_pairwise_distance > 0 else 0.0
            )
        else:
            avg_pairwise_distance = 0.0
            density = 0.0

        # Get cluster persistence from HDBSCAN (if available)
        # This indicates how stable/prominent the cluster is
        persistence = None
        if clusterer and hasattr(clusterer, "cluster_persistence_"):
            try:
                # Try to extract numeric ID from label for persistence lookup
                if label.replace(".", "").replace("-", "").isdigit():
                    # For hierarchical labels like "2.0", use the last numeric part
                    numeric_parts = [int(x) for x in label.split(".") if x.isdigit()]
                    if numeric_parts and numeric_parts[-1] < len(
                        clusterer.cluster_persistence_
                    ):
                        persistence = float(
                            clusterer.cluster_persistence_[numeric_parts[-1]]
                        )
            except Exception:
                pass

        stats = {
            "cluster_id": label,  # Keep as string to support hierarchical labels
            "size": cluster_size,
            "percentage": float(cluster_size / n_samples * 100),
            "compactness": compactness,
            "std_distance": std_distance,
            "max_distance": max_distance,
            "density": density,
            "avg_pairwise_distance": avg_pairwise_distance,
            "persistence": persistence,
        }

        cluster_stats.append(stats)

    # Sort by cluster size (descending)
    cluster_stats.sort(key=lambda x: x["size"], reverse=True)

    return {
        "overall": overall_stats,
        "clusters": cluster_stats,
    }


def create_2d_visualization(embeddings_2d, labels, save_path: str = "clusters_2d.png"):
    """Create a 2D scatter plot of clusters."""
    plt.figure(figsize=(14, 10))

    labels = np.asarray(labels)
    noise = noise_mask(labels)
    clusters = sorted(set(labels[~noise].tolist()))
    n_clusters = len(clusters)

    if noise.any():
        plt.scatter(
            embeddings_2d[noise, 0],
            embeddings_2d[noise, 1],
            c="lightgrey",
            alpha=0.3,
            s=20,
            label="Noise",
        )

    # Plot each cluster with a different color
    for label in clusters:
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            alpha=0.6,
            s=30,
            label=f"Cluster {label} ({mask.sum()} points)",
        )

    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.title(
        f"Cluster Visualization (2D)\n{n_clusters} clusters, {noise.sum()} noise points",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    # Only show legend if not too many clusters
    if n_clusters < 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 2D visualization to {save_path}")


def create_3d_visualization(embeddings_3d, labels, save_path: str = "clusters_3d.png"):
    """Create a 3D scatter plot of clusters."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    labels = np.asarray(labels)
    noise = noise_mask(labels)
    clusters = sorted(set(labels[~noise].tolist()))
    n_clusters = len(clusters)

    if noise.any():
        ax.scatter(
            embeddings_3d[noise, 0],
            embeddings_3d[noise, 1],
            embeddings_3d[noise, 2],
            c="lightgrey",
            alpha=0.3,
            s=20,
            label="Noise",
        )

    # Plot each cluster with a different color
    for label in clusters:
        mask = labels == label
        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            alpha=0.6,
            s=30,
            label=f"Cluster {label} ({mask.sum()} pts)",
        )

    ax.set_xlabel("UMAP Component 1", fontsize=10)
    ax.set_ylabel("UMAP Component 2", fontsize=10)
    ax.set_zlabel("UMAP Component 3", fontsize=10)
    ax.set_title(
        f"Cluster Visualization (3D)\n{n_clusters} clusters, {noise.sum()} noise points",
        fontsize=14,
    )

    # Only show legend if not too many clusters
    if n_clusters < 15:
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 3D visualization to {save_path}")


def save_cluster_statistics(stats, save_path: str | Path = "cluster_stats.json"):
    """Save cluster statistics to a JSON file."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved cluster statistics to {save_path}")


def should_recluster(cluster_stats: dict, config: HierarchicalConfig) -> bool:
    """
    Determine if a cluster needs refinement.

    Returns True if cluster is both large and vague.
    """
    return cluster_stats["density"] < config.density_threshold


def extract_cluster_embeddings(
    embeddings: np.ndarray, labels: np.ndarray, cluster_id: str | int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract original embeddings and indices for a specific cluster.

    Handles both numeric (0, 1, 2) and hierarchical ("2.0", "2.1") labels.
    Returns: (cluster_embeddings, cluster_indices)
    """
    # Always convert to strings for consistent comparison
    cluster_id_str = str(cluster_id)
    mask = np.array([str(label) == cluster_id_str for label in labels])

    indices = np.where(mask)[0]
    cluster_embeddings = embeddings[indices]

    logger.debug(f"Extracted {len(indices)} points for cluster '{cluster_id_str}'")

    return cluster_embeddings, indices


def refine_cluster(
    original_embeddings: np.ndarray,
    cluster_indices: np.ndarray,
    parent_label: str,
    config: HierarchicalConfig,
    depth: int,
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

        # Extract cluster embeddings
        _cluster_embeddings, cluster_indices = extract_cluster_embeddings(
            original_embeddings, current_labels, cluster_id_str
        )
        logger.debug(f"  Extracted {len(cluster_indices)} points for refinement")

        # Attempt refinement
        sub_labels, sub_reduced = refine_cluster(
            original_embeddings, cluster_indices, cluster_id_str, config, depth
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
