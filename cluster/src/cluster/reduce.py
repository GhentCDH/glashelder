import logging

import numpy as np
from hdbscan import HDBSCAN

# installed packages
from umap import UMAP

logger = logging.getLogger(__name__)


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
