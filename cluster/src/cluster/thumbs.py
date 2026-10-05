"""Sample images of a cluster, and small versions of them, for the viewer and the report."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from common.images import is_url, load

from .result import Clustering


def samples(clustering: Clustering, label: int, n: int) -> list[int]:
    """The `n` most typical rows of a cluster: the ones with the lowest outlier score."""
    members = np.flatnonzero(clustering.labels == label)
    order = np.argsort(clustering.outlier_scores[members], kind="stable")
    return members[order[:n]].tolist()


def thumbnail(ref: str | Path, size: int, quality: int = 80) -> bytes:
    """A JPEG of at most `size` x `size` pixels."""
    if is_url(ref):
        image = load(ref)
    else:
        image = Image.open(ref)
        # a JPEG decodes straight at 1/2, 1/4 or 1/8 scale: much less to read
        image.draft("RGB", (size, size))
        image = image.convert("RGB")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()
