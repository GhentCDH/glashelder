import logging

import numpy as np
import rerun as rr
from PIL import Image

RERUN_ID = "clipcluster"

logger = logging.getLogger(__name__)


def init_rerun(address: str | None):
    if address:
        rr.init(RERUN_ID, spawn=False)
        if address.startswith("rerun+http"):
            # connect to runnin rerun instance
            rr.connect_grpc(address)
        else:
            # save to a file
            rr.save(address)

    else:
        rr.init(RERUN_ID, spawn=True)


def log_reduced(
    latent: np.ndarray, count: int | None = None, base_path: str = "latent"
):
    """
    Log 3D points to Rerun

    Args:
        latent: The reduced embeddings, shape (N, D) where D >= 3.
        count: Number of points to sample. If None, logs all points.
        base_path: The base Rerun entity path. Clusters logged as {base_path}/{cluster_id}.
    """
    n_points = len(latent)

    if count is not None and count < n_points:
        indices = np.random.choice(n_points, size=count, replace=False)
        latent = latent[indices]

    rr.log(base_path, rr.Points3D(latent, radii=0.02))


def log_clustered_points(
    latent: np.ndarray,
    labels: np.ndarray,
    count: int | None = None,
    base_path: str = "clusters",
):
    """
    Log 3D points to Rerun, with each cluster on a separate entity path.

    Args:
        latent: The reduced embeddings, shape (N, D) where D >= 3.
        labels: Cluster labels for each point, shape (N,). -1 indicates noise.
        count: Number of points to sample. If None, logs all points.
        base_path: The base Rerun entity path. Clusters logged as {base_path}/{cluster_id}.
    """
    n_points = len(latent)

    if count is not None and count < n_points:
        indices = np.random.choice(n_points, size=count, replace=False)
        latent = latent[indices]
        labels = labels[indices]

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    cmap = _generate_colormap(n_clusters)

    for label in unique_labels:
        mask = labels == label
        positions = latent[mask, :3]

        if label == -1:
            path = f"{base_path}/noise"
            color = [128, 128, 128]
        else:
            path = f"{base_path}/{label}"
            color = cmap[label % len(cmap)].tolist()

        rr.log(path, rr.Points3D(positions, colors=color, radii=0.02))


def log_clustered_points_density(
    latent: np.ndarray,
    labels: np.ndarray,
    stats: dict,
    count: int | None = None,
    base_path: str = "clusters",
):
    """
    Log 3D points to Rerun, with each cluster on a separate entity path.

    Args:
        latent: The reduced embeddings, shape (N, D) where D >= 3.
        labels: Cluster labels for each point, shape (N,). -1 indicates noise.
        count: Number of points to sample. If None, logs all points.
        base_path: The base Rerun entity path. Clusters logged as {base_path}/{cluster_id}.
    """
    n_points = len(latent)

    if count is not None and count < n_points:
        indices = np.random.choice(n_points, size=count, replace=False)
        latent = latent[indices]
        labels = labels[indices]

    unique_labels = np.unique(labels)
    cmap = _generate_density_colormap(unique_labels, stats, "plasma")

    for label in unique_labels:
        mask = labels == label
        positions = latent[mask, :3]

        if label == -1:
            path = f"{base_path}/noise"
            color = [128, 128, 128]
        else:
            path = f"{base_path}/{label}"
            color = [cmap[label]]

        rr.log(path, rr.Points3D(positions, colors=color, radii=0.02))


def log_image_3d(
    position: np.ndarray,
    image_path: str,
    entity_path: str,
    size: float = 0.5,
):
    """
    Log an image as a textured rectangle at a 3D position in Rerun.

    Args:
        position: 3D coordinates [x, y, z] where the image should appear.
        image_path: Path to the image file.
        entity_path: The Rerun entity path for this image.
        size: Half-width/height of the image rectangle in 3D space.
    """

    img = Image.open(image_path).convert("RGB")
    ratio = 150 / img.width
    new_height = int(img.height * ratio)
    img = img.resize((150, new_height), Image.Resampling.NEAREST)
    img_array = np.array(img)

    # Create a flat rectangle at the position, facing the camera (z-axis)
    x, y, z = position[:3]
    half = size

    # Define rectangle corners (in XY plane, centered at position)
    positions = np.array(
        [
            [x - half, y - half, z],
            [x + half, y - half, z],
            [x + half, y + half, z],
            [x - half, y + half, z],
        ],
        dtype=np.float32,
    )

    # Two triangles to form a quad
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    # UV coordinates for texture mapping
    texcoords = np.array(
        [
            [0.0, 1.0],  # bottom-left
            [1.0, 1.0],  # bottom-right
            [1.0, 0.0],  # top-right
            [0.0, 0.0],  # top-left
        ],
        dtype=np.float32,
    )

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=positions,
            triangle_indices=indices,
            vertex_texcoords=texcoords,
            albedo_texture=img_array,
        ),
    )


def log_images_3d(
    latent: np.ndarray,
    image_paths: list[str],
    count: int,
    base_path: str = "images",
    size: float = 0.3,
):
    """
    Log `count` random images at their 3D positions in Rerun.

    Args:
        latent: The reduced embeddings, shape (N, D) where D >= 3.
        image_paths: List of paths to images, corresponding to latent rows.
        count: Number of random images to log.
        base_path: The base Rerun entity path.
        size: Size of each image in 3D space.
    """
    n_points = len(latent)
    count = min(count, n_points)

    indices = np.random.choice(n_points, size=count, replace=False)

    for i, idx in enumerate(indices):
        position = latent[idx]
        image_path = image_paths[idx]
        entity_path = f"{base_path}/{i}"

        log_image_3d(position, image_path, entity_path, size=size)


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


def _generate_density_colormap(
    labels, stats, cmap_name: str = "plasma"
) -> dict[str, list[int]]:
    """Generate n distinct colours from a matplotlib colormap."""
    import matplotlib.pyplot as plt

    colors: dict[str, list[int]] = {}

    cmap = plt.get_cmap(cmap_name)
    max_density: float = max([stat["density"] for stat in stats["clusters"]])

    def _cmap(x: float) -> list[int]:
        rgba = cmap(x)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return [r, g, b]

    def _get_color(label) -> list[int]:
        cluster = next(
            (str(stat["cluster_id"]) == str(label) for stat in stats["clusters"]), None
        )

        if cluster is None:
            logger.warning(f"Couldn't find stats for cluster {label}")
            return _cmap(0)

        density = cluster["density"]

        return _cmap(density / max_density)

    for label in labels:
        colors[label] = _get_color(label)

    return colors


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
