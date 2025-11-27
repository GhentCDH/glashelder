from typing import Optional

import numpy as np
import rerun as rr

RERUN_ID = "clipcluster"


def init_rerun(address: Optional[str]):
    if address:
        rr.init(RERUN_ID, spawn=False)
        rr.connect_grpc(address)
    else:
        rr.init(RERUN_ID, spawn=True)


def log_clustered_points(
    latent: np.ndarray,
    labels: np.ndarray,
    count: Optional[int] = None,
    entity_path: str = "clusters/points",
):
    """
    Log 3D points to Rerun, coloured by cluster label.

    Args:
        latent: The reduced embeddings, shape (N, D) where D >= 3.
        labels: Cluster labels for each point, shape (N,). -1 indicates noise.
        count: Number of points to sample. If None, logs all points.
        entity_path: The Rerun entity path to log to.
    """
    n_points = len(latent)

    if count is not None and count < n_points:
        indices = np.random.choice(n_points, size=count, replace=False)
        latent = latent[indices]
        labels = labels[indices]

    positions = latent[:, :3]

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    cmap = _generate_colormap(n_clusters)

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label == -1:
            colors[i] = [128, 128, 128]  # grey for noise
        else:
            colors[i] = cmap[label % len(cmap)]

    rr.log(entity_path, rr.Points3D(positions, colors=colors, radii=0.02))


def _generate_colormap(n: int) -> np.ndarray:
    """Generate n distinct colours using HSV."""
    if n == 0:
        return np.array([[255, 0, 0]], dtype=np.uint8)

    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    return v, v, v
