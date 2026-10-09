import logging

import matplotlib.pyplot as plt
import numpy as np

from .stats import noise_mask

logger = logging.getLogger(__name__)


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
