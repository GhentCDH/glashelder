"""ALIKED features of a whole collection, made once: `STORE/features/<collection>/`.

    meta.json        matcher, resize and max_keypoints the features were made with
    refs.txt         one image per line, relative to the collection's image directory
    counts.bin       int32 per image: how many keypoints it has
    shapes.bin       int32 (height, width) per image, after resizing
    keypoints.bin    float32 (max_keypoints, 2) per image, unused slots zero
    descriptors.bin  float16 (max_keypoints, 128) per image, unused slots zero

Every image takes the same room, so image i is at a fixed place in each file and
the files are memory mapped: reading the features of 20 candidates costs 20 small
reads, not an image decode and an ALIKED pass each. That is about 0.5 MB per
image at 2048 keypoints.

A batch goes to the data files first and to refs.txt last, so refs.txt says which
rows are whole. Opening for writing cuts the data files back to the rows in
refs.txt: a run that stops loses at most the batch in progress.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.errors import UserError
from common.fs import atomic_write

from .match import Settings
from .store import MATCHER

FEATURES = "features"
META, REFS = "meta.json", "refs.txt"
DIM = 128
"""Length of an ALIKED descriptor."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Layout:
    """The files of a feature store, and how much room one image takes in each."""

    max_keypoints: int

    @property
    def files(self) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
        return {
            "counts.bin": (np.dtype("<i4"), ()),
            "shapes.bin": (np.dtype("<i4"), (2,)),
            "keypoints.bin": (np.dtype("<f4"), (self.max_keypoints, 2)),
            "descriptors.bin": (np.dtype("<f2"), (self.max_keypoints, DIM)),
        }

    def row_bytes(self, name: str) -> int:
        dtype, shape = self.files[name]
        return dtype.itemsize * int(np.prod(shape, dtype=int))


def location(store: Path, collection: str) -> Path:
    return Path(store) / FEATURES / collection


def describe(settings: Settings) -> dict:
    return {
        "matcher": MATCHER,
        "resize": settings.resize,
        "max_keypoints": settings.max_keypoints,
    }


def read_refs(path: Path) -> list[str]:
    if not (path / REFS).is_file():
        return []
    return (path / REFS).read_text().splitlines()


class Writer:
    """Appends the features of new images to the store at `path`."""

    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.layout = Layout(settings.max_keypoints)
        made = describe(settings)
        if (path / META).is_file():
            stored = json.loads((path / META).read_text())
            if stored != made:
                raise UserError(
                    f"{path} holds features made with {stored}, not {made}. Remove "
                    "that directory to make them again with these settings"
                )
        else:
            path.mkdir(parents=True, exist_ok=True)
            atomic_write(path / META, json.dumps(made, indent=2) + "\n")
        self.refs = read_refs(path)
        self.known = set(self.refs)
        for name in self.layout.files:
            with open(path / name, "ab") as file:
                file.truncate(len(self.refs) * self.layout.row_bytes(name))

    def add(self, rows: list[tuple[str, dict, tuple[int, int]]]) -> None:
        """Commit `(ref, features, (height, width))` of a batch of images, with
        the features as `Matcher.extract` gives them."""
        if not rows:
            return
        keypoints_max = self.layout.max_keypoints
        counts = np.zeros(len(rows), dtype="<i4")
        shapes = np.zeros((len(rows), 2), dtype="<i4")
        keypoints = np.zeros((len(rows), keypoints_max, 2), dtype="<f4")
        descriptors = np.zeros((len(rows), keypoints_max, DIM), dtype="<f2")
        for i, (_, features, shape) in enumerate(rows):
            points = features["keypoints"][0].cpu().numpy()
            n = min(len(points), keypoints_max)
            counts[i], shapes[i] = n, shape
            keypoints[i, :n] = points[:n]
            descriptors[i, :n] = features["descriptors"][0, :n].half().cpu().numpy()
        data = {
            "counts.bin": counts,
            "shapes.bin": shapes,
            "keypoints.bin": keypoints,
            "descriptors.bin": descriptors,
        }
        for name, array in data.items():
            with open(self.path / name, "ab") as file:
                file.write(array.tobytes())
                file.flush()
                os.fsync(file.fileno())
        keys = [ref for ref, _, _ in rows]
        with open(self.path / REFS, "a") as file:
            file.write("".join(f"{key}\n" for key in keys))
        self.refs.extend(keys)
        self.known.update(keys)


class Features:
    """The features of a collection, read without loading them into memory."""

    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.refs = read_refs(path)
        self.rows = {ref: row for row, ref in enumerate(self.refs)}
        layout = Layout(settings.max_keypoints)
        n = len(self.refs)
        self.arrays = {
            name.removesuffix(".bin"): np.memmap(
                path / name, dtype=dtype, mode="r", shape=(n, *shape)
            )
            for name, (dtype, shape) in layout.files.items()
        }

    @classmethod
    def find(cls, path: Path, settings: Settings) -> "Features | None":
        """The features at `path`, or None when there are none for `settings`."""
        if not (path / META).is_file():
            logger.info(f"no precomputed features in {path}")
            return None
        stored = json.loads((path / META).read_text())
        if stored != describe(settings):
            logger.warning(
                f"{path} holds features made with {stored}, not "
                f"{describe(settings)}; they are not used"
            )
            return None
        if not read_refs(path):
            return None
        features = cls(path, settings)
        logger.info(f"{path}: precomputed features of {len(features):,} images")
        return features

    def __len__(self) -> int:
        return len(self.refs)

    def get(self, row: int, device: str) -> tuple[dict, tuple[int, int]]:
        """The features of image `row` as LightGlue takes them, and its
        (height, width)."""
        n = int(self.arrays["counts"][row])
        height, width = (int(v) for v in self.arrays["shapes"][row])
        keypoints = torch.from_numpy(np.array(self.arrays["keypoints"][row, :n]))
        descriptors = torch.from_numpy(np.array(self.arrays["descriptors"][row, :n]))
        features = {
            "keypoints": keypoints[None].to(device),
            "descriptors": descriptors[None].to(device).float(),
            "image_size": torch.tensor([[width, height]], device=device).float(),
        }
        return features, (height, width)
