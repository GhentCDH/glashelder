import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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


def save_cluster_statistics(stats, save_path: str | Path = "cluster_stats.json"):
    """Save cluster statistics to a JSON file."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved cluster statistics to {save_path}")
