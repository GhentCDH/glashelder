import logging

# installed packages
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

# local source

logger = logging.getLogger(__name__)


def reduce_dimensions(embeddings, dimensions: int = 3):
    umap_model = UMAP(n_components=dimensions, n_neighbors=20, min_dist=0.01)
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
