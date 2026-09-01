"""Image loading and SIFT feature helpers shared by the matchers."""

import cv2
import numpy as np


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


def load_image(ref: str) -> np.ndarray:
    """Load an image, from disk or from a URL, as an OpenCV BGR array.

    Every feature extractor goes through here, so a reference scraped as a URL
    works anywhere a path does.
    """
    from common.images import is_url, load

    if is_url(ref):
        return cv2.cvtColor(np.array(load(ref)), cv2.COLOR_RGB2BGR)

    img = cv2.imread(ref)
    if img is None:
        raise ValueError(f"Could not load image {ref}")
    return img
