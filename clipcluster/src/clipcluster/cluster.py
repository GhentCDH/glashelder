import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN

# installed packages
from umap import UMAP

# local source

logger = logging.getLogger(__name__)


def reduce_dimensions(embeddings, dimensions: int = 30):
    umap_model = UMAP(n_components=dimensions, n_neighbors=15, min_dist=0.05)
    return umap_model.fit_transform(embeddings)


def create_clusterer(latent, min_cluster_size: int = 50):
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(latent)

    return clusterer


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
        labels: Cluster labels from HDBSCAN
        clusterer: The fitted HDBSCAN clusterer object

    Returns:
        Dictionary containing overall and per-cluster statistics
    """
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_samples = len(labels)
    n_noise = sum(labels == -1)

    # Overall statistics
    overall_stats = {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "n_noise": n_noise,
        "noise_ratio": n_noise / n_samples if n_samples > 0 else 0,
    }

    # Calculate silhouette score if we have more than 1 cluster and less than all noise
    if n_clusters > 1 and n_noise < n_samples:
        try:
            from sklearn.metrics import silhouette_score

            # Only calculate silhouette for non-noise points
            non_noise_mask = labels != -1
            if sum(non_noise_mask) > 0:
                silhouette = silhouette_score(
                    embeddings[non_noise_mask], labels[non_noise_mask]
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
        if label == -1:
            continue  # Skip noise in per-cluster stats

        cluster_mask = labels == label
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
            density = 1.0 / avg_pairwise_distance if avg_pairwise_distance > 0 else 0
        else:
            avg_pairwise_distance = 0
            density = 0

        # Get cluster persistence from HDBSCAN (if available)
        # This indicates how stable/prominent the cluster is
        persistence = None
        if hasattr(clusterer, "cluster_persistence_"):
            try:
                cluster_id = label
                if cluster_id < len(clusterer.cluster_persistence_):
                    persistence = float(clusterer.cluster_persistence_[cluster_id])
            except Exception:
                pass

        stats = {
            "cluster_id": int(label),
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

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Plot each cluster with a different color
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise points in grey
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c="lightgrey",
                alpha=0.3,
                s=20,
                label="Noise",
            )
        else:
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                alpha=0.6,
                s=30,
                label=f"Cluster {label} ({sum(mask)} points)",
            )

    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.title(
        f"Cluster Visualization (2D)\n{n_clusters} clusters, {sum(labels == -1)} noise points",
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

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Plot each cluster with a different color
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise points in grey
            ax.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c="lightgrey",
                alpha=0.3,
                s=20,
                label="Noise",
            )
        else:
            ax.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                alpha=0.6,
                s=30,
                label=f"Cluster {label} ({sum(mask)} pts)",
            )

    ax.set_xlabel("UMAP Component 1", fontsize=10)
    ax.set_ylabel("UMAP Component 2", fontsize=10)
    ax.set_zlabel("UMAP Component 3", fontsize=10)
    ax.set_title(
        f"Cluster Visualization (3D)\n{n_clusters} clusters, {sum(labels == -1)} noise points",
        fontsize=14,
    )

    # Only show legend if not too many clusters
    if n_clusters < 15:
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 3D visualization to {save_path}")


def save_cluster_statistics(stats, save_path: str = "cluster_stats.json"):
    """Save cluster statistics to a JSON file."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved cluster statistics to {save_path}")
