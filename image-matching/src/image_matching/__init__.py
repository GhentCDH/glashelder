"""Image matching and artwork recognition using SIFT features."""

from image_matching.recognition import (
    MatchResult,
    RecognitionDatabase,
    build_database,
    find_painting,
)
from image_matching.stitching import stitch_images

__all__ = [
    "MatchResult",
    "RecognitionDatabase",
    "build_database",
    "find_painting",
    "stitch_images",
]
