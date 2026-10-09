"""Sample images of a cluster, and small versions of them, for the viewer and the report."""

import io
from pathlib import Path

import numpy as np

from common import images

from .result import Clustering


def samples(clustering: Clustering, label: int, n: int) -> list[int]:
    """The `n` most typical rows of a cluster: the ones with the lowest outlier score."""
    members = np.flatnonzero(clustering.labels == label)
    order = np.argsort(clustering.outlier_scores[members], kind="stable")
    return members[order[:n]].tolist()


def thumbnail(ref: str | Path, size: int, quality: int = 80) -> bytes:
    """A JPEG of at most `size` x `size` pixels."""
    buffer = io.BytesIO()
    images.thumbnail(ref, size).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
