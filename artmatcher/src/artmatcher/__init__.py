"""Matching glass plate photos against a reference set of artworks."""

from .ml_classifier import FeatureExtractor, MLClassifier
from .stitching import compute_homography, extract_features, load_image, resize_max

__all__ = [
    "FeatureExtractor",
    "MLClassifier",
    "compute_homography",
    "extract_features",
    "load_image",
    "resize_max",
]
