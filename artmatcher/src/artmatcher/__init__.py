"""Matching glass plate photos against a reference set of artworks."""

from .classifier import MLClassifier
from .stitching import compute_homography, extract_features, load_image, resize_max

__all__ = [
    "MLClassifier",
    "compute_homography",
    "extract_features",
    "load_image",
    "resize_max",
]
