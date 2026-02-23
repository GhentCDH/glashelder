"""Image matching and artwork recognition using SIFT features."""

from .recognition import (
    MatchResult,
    RecognitionDatabase,
    build_database,
    find_painting,
)
from .stitching import stitch_images

__all__ = [
    "MatchResult",
    "RecognitionDatabase",
    "build_database",
    "find_painting",
    "stitch_images",
]
