"""Core image stitching logic using SIFT features and homography."""

import gc
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StitchResult:
    """Result of a stitching operation."""

    image: np.ndarray
    base_index: int
    stitched_count: int
    skipped_stretch: list[int]
    skipped_no_connection: list[int]


def resize_max(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image so its largest dimension is at most max_dim."""
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def extract_features(image: np.ndarray, sift) -> tuple:
    """Extract SIFT keypoints and descriptors from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def find_matches(des1, des2, bf) -> list:
    """Find good matches between two sets of descriptors using ratio test."""
    if des1 is None or des2 is None:
        return []
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


def compute_homography(kp1, kp2, matches) -> np.ndarray | None:
    """Compute homography matrix from matched keypoints."""
    if len(matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # type:ignore
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # type:ignore
    h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if h is None:
        return None
    # Count inliers
    inliers = mask.ravel().sum() if mask is not None else 0
    if inliers < 4:
        return None
    return h


def compute_warp_stretch(h: np.ndarray, img_shape: tuple[int, int]) -> float:
    """Compute how much a homography stretches/distorts an image.

    Returns the maximum ratio of (warped edge length / original edge length)
    across all four edges. A value of 1.0 means no stretch, higher values
    mean more distortion.
    """
    img_h, img_w = img_shape[:2]
    corners = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]).reshape(  # type:ignore
        -1, 1, 2
    )
    warped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)

    # Original edge lengths
    orig_edges = [
        img_w,  # top
        img_h,  # right
        img_w,  # bottom
        img_h,  # left
    ]

    # Warped edge lengths
    warped_edges = []
    for i in range(4):
        p1 = warped[i]
        p2 = warped[(i + 1) % 4]
        warped_edges.append(np.linalg.norm(p2 - p1))

    # Compute stretch ratios
    ratios = []
    for orig, warped_len in zip(orig_edges, warped_edges, strict=True):
        if orig > 0:
            ratios.append(max(warped_len / orig, orig / warped_len))

    return max(ratios) if ratios else 1.0


def stitch_pair(
    img1: np.ndarray, img2: np.ndarray, h: np.ndarray, max_canvas: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch img1 onto img2's coordinate system using homography h.

    Returns the stitched image and the updated homography to map
    from original img2 coords to stitched canvas coords.

    If max_canvas is set, scales down the result if it exceeds that dimension.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute bounding box of warped img1
    corners_1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)  # type:ignore
    warped_corners = cv2.perspectiveTransform(corners_1, h)

    all_corners = np.concatenate(
        [
            warped_corners,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2),  # type:ignore
        ]
    )

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())  # type:ignore
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())  # type:ignore

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Compute scale factor if canvas exceeds max size
    scale = 1.0
    if max_canvas is not None and max(canvas_w, canvas_h) > max_canvas:
        scale = max_canvas / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    # Translation to shift everything into positive coordinates (and scale if needed)
    translation = np.array(
        [[scale, 0, -x_min * scale], [0, scale, -y_min * scale], [0, 0, 1]],
        dtype=np.float64,
    )

    canvas_size = (canvas_w, canvas_h)
    stitched = cv2.warpPerspective(img1, translation @ h, canvas_size)

    # Place img2 on the canvas (scaled)
    if scale < 1.0:
        img2_scaled = cv2.resize(
            img2,
            (int(w2 * scale), int(h2 * scale)),
            interpolation=cv2.INTER_AREA,
        )
        y_off = int(-y_min * scale)
        x_off = int(-x_min * scale)
        sh2, sw2 = img2_scaled.shape[:2]
    else:
        img2_scaled = img2
        y_off = -y_min
        x_off = -x_min
        sh2, sw2 = h2, w2

    # Create a mask for img2 region
    mask = stitched[y_off : y_off + sh2, x_off : x_off + sw2] == 0
    stitched[y_off : y_off + sh2, x_off : x_off + sw2] = np.where(
        mask, img2_scaled, stitched[y_off : y_off + sh2, x_off : x_off + sw2]
    )

    return stitched, translation


def load_image(path: str) -> np.ndarray:
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image {path}")
    return img


def create_flann_matcher():
    """Create a FLANN-based matcher (much faster than BFMatcher for large sets)."""
    # FLANN parameters for SIFT
    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE = 1
    search_params = {"checks": 50}
    return cv2.FlannBasedMatcher(index_params, search_params)


def find_matches_flann(des1, des2, flann) -> list:
    """Find good matches using FLANN with ratio test."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    # FLANN requires float32 descriptors
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    return good_matches


def stitch_images(
    image_paths: list[str],
    *,
    feature_size: int | None = None,
    max_canvas: int | None = 8000,
    max_stretch: float = 3.0,
    on_progress: Callable[[str], None] | None = None,
) -> StitchResult:
    """Stitch multiple images together into a panorama.

    Args:
        image_paths: List of paths to images to stitch.
        feature_size: Max size for feature extraction (None for full resolution).
        max_canvas: Max canvas dimension (scales down if exceeded).
        max_stretch: Max allowed stretch ratio for warped images.
        on_progress: Optional callback for progress messages.

    Returns:
        StitchResult with the stitched image and metadata.

    Raises:
        ValueError: If fewer than 2 images provided or no matches found.
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    n = len(image_paths)
    log(f"Processing {n} images...")

    # Phase 1: Extract features from all images
    log("Extracting features...")
    sift = cv2.SIFT_create()  # type:ignore

    features: list[tuple] = []
    scales: list[float] = []
    original_shapes: list[tuple[int, int]] = []

    for i, path in enumerate(image_paths):
        full_img = load_image(path)
        original_shapes.append((full_img.shape[0], full_img.shape[1]))

        if feature_size is not None:
            img = resize_max(full_img, feature_size)
            scale = img.shape[1] / full_img.shape[1]
        else:
            img = full_img
            scale = 1.0

        kp, des = extract_features(img, sift)
        features.append((kp, des))
        scales.append(scale)
        del img, full_img

        if (i + 1) % 10 == 0:
            log(f"  Extracted features from {i + 1}/{n} images")
            gc.collect()

    log(f"  Extracted features from all {n} images")
    gc.collect()

    # Phase 2: Find matches
    log("Finding matches between image pairs...")
    flann = create_flann_matcher()

    match_counts: dict[tuple[int, int], int] = {}
    homographies: dict[tuple[int, int], np.ndarray] = {}
    connected_to: dict[int, set[int]] = {i: set() for i in range(n)}

    pairs_checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            kp_i, des_i = features[i]
            kp_j, des_j = features[j]

            matches_ij = find_matches_flann(des_i, des_j, flann)
            matches_ji = find_matches_flann(des_j, des_i, flann)

            match_counts[(i, j)] = len(matches_ij)
            match_counts[(j, i)] = len(matches_ji)
            pairs_checked += 1

            if len(matches_ij) >= 4:
                h = compute_homography(kp_i, kp_j, matches_ij)
                if h is not None:
                    s_i = np.array(
                        [[scales[i], 0, 0], [0, scales[i], 0], [0, 0, 1]],
                        dtype=np.float64,
                    )
                    s_j_inv = np.array(
                        [[1 / scales[j], 0, 0], [0, 1 / scales[j], 0], [0, 0, 1]],
                        dtype=np.float64,
                    )
                    homographies[(i, j)] = s_j_inv @ h @ s_i
                    connected_to[i].add(j)
                    connected_to[j].add(i)

            if len(matches_ji) >= 4:
                h = compute_homography(kp_j, kp_i, matches_ji)
                if h is not None:
                    s_j = np.array(
                        [[scales[j], 0, 0], [0, scales[j], 0], [0, 0, 1]],
                        dtype=np.float64,
                    )
                    s_i_inv = np.array(
                        [[1 / scales[i], 0, 0], [0, 1 / scales[i], 0], [0, 0, 1]],
                        dtype=np.float64,
                    )
                    homographies[(j, i)] = s_i_inv @ h @ s_j

        if (i + 1) % 10 == 0:
            log(f"  Processed pairs for {i + 1}/{n} images ({pairs_checked} pairs)")

    log(f"  Checked {pairs_checked} pairs (vs {n * (n - 1) // 2} exhaustive)")

    # Find best base image
    connectivity = [
        sum(match_counts.get((i, j), 0) for j in connected_to[i]) for i in range(n)
    ]

    if max(connectivity) == 0:
        raise ValueError("Could not find enough matches between images")

    base_idx = connectivity.index(max(connectivity))
    log(f"Using image {base_idx + 1} as base")

    # Build stitching order using BFS, filtering excessive warping
    stitched_indices = {base_idx}
    pending = set(range(n)) - {base_idx}
    skipped_stretch: list[int] = []
    skipped_no_connection: list[int] = []
    stitch_order: list[tuple[int, int]] = []

    canvas_h_bfs: dict[int, np.ndarray] = {base_idx: np.eye(3, dtype=np.float64)}

    while pending:
        best_pair = None
        best_matches = 0
        best_h_to_canvas = None

        for p in pending:
            for s in stitched_indices:
                if (p, s) in homographies:
                    h_p_to_s = homographies[(p, s)]
                    h_s_to_canvas = canvas_h_bfs[s]
                    h_p_to_canvas = h_s_to_canvas @ h_p_to_s

                    stretch = compute_warp_stretch(h_p_to_canvas, original_shapes[p])
                    if stretch > max_stretch:
                        continue

                    if match_counts[(p, s)] > best_matches:
                        best_matches = match_counts[(p, s)]
                        best_pair = (p, s)
                        best_h_to_canvas = h_p_to_canvas

        if best_pair is None:
            remaining_with_connections = [
                p
                for p in pending
                if any((p, s) in homographies for s in stitched_indices)
            ]
            if remaining_with_connections:
                log(
                    f"Skipping {len(remaining_with_connections)} images (excessive warping)"
                )
                skipped_stretch.extend(remaining_with_connections)
                pending -= set(remaining_with_connections)
                continue
            if pending:
                log(f"Warning: Could not connect {len(pending)} images")
                skipped_no_connection.extend(pending)
            break

        stitch_order.append(best_pair)
        stitched_indices.add(best_pair[0])
        pending.remove(best_pair[0])
        canvas_h_bfs[best_pair[0]] = best_h_to_canvas  # type:ignore

    del features
    gc.collect()

    log(f"Stitching {len(stitch_order) + 1} images...")

    # Phase 3: Perform stitching
    result = load_image(image_paths[base_idx])
    canvas_h = {base_idx: np.eye(3, dtype=np.float64)}

    for step, (src_idx, dst_idx) in enumerate(stitch_order):
        h_src_to_dst = homographies[(src_idx, dst_idx)]
        h_dst_to_canvas = canvas_h[dst_idx]
        h_src_to_canvas = h_dst_to_canvas @ h_src_to_dst

        stretch = compute_warp_stretch(h_src_to_canvas, original_shapes[src_idx])
        log(
            f"  Step {step + 1}/{len(stitch_order)}: Adding image {src_idx + 1} (stretch {stretch:.1f})"
        )

        src_img = load_image(image_paths[src_idx])
        result, translation = stitch_pair(src_img, result, h_src_to_canvas, max_canvas)

        del src_img
        gc.collect()

        for idx in canvas_h:
            canvas_h[idx] = translation @ canvas_h[idx]
        canvas_h[src_idx] = translation @ h_src_to_canvas

    return StitchResult(
        image=result,
        base_index=base_idx,
        stitched_count=len(stitch_order) + 1,
        skipped_stretch=skipped_stretch,
        skipped_no_connection=skipped_no_connection,
    )
