"""Local feature matching of a plate and a reference photo, and the geometry check.

Both images are made grayscale and resized to the same long side, so a colour
photo and a glass plate look alike to the detector, and pixel thresholds mean the
same thing for every pair. ALIKED finds the keypoints and LightGlue matches them.

The check is a fundamental matrix: the same building from another viewpoint is
not a plane, so a homography only fits one facade. The homography inliers are
kept as well, as a sign of "almost the same view".
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from common.images import load


@dataclass(frozen=True)
class Settings:
    resize: int = 1024
    """Long side of both images, in pixels."""
    max_keypoints: int = 2048
    f_px: float = 4.0
    """Fundamental matrix inlier threshold, in pixels at `resize`."""
    h_px: float = 8.0
    """Homography inlier threshold, in pixels at `resize`."""
    grid: int = 8
    """`cells` counts the cells of a grid x grid grid that hold an inlier."""


def prepare(ref: str, resize: int) -> np.ndarray:
    """`ref` as a grayscale image with a long side of `resize`."""
    return grayscale(load(ref), resize)


def grayscale(picture: Image.Image, resize: int) -> np.ndarray:
    """`picture` as a grayscale array with a long side of `resize`."""
    image = np.asarray(picture.convert("L"))
    scale = resize / max(image.shape)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


class Matcher:
    """ALIKED features and LightGlue matching, on one device."""

    def __init__(self, settings: Settings, device: str):
        from lightglue import ALIKED, LightGlue

        self.device = device
        self.extractor = (
            ALIKED(max_num_keypoints=settings.max_keypoints).eval().to(device)
        )
        self.matcher = LightGlue(features="aliked").eval().to(device)

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> dict:
        tensor = torch.from_numpy(image).to(self.device).float().div(255)
        return self.extractor.extract(tensor[None].repeat(3, 1, 1), resize=None)

    @staticmethod
    def compact(features: dict) -> dict:
        """`features` with float16 descriptors: half the memory, to keep many."""
        return {**features, "descriptors": features["descriptors"].half()}

    @staticmethod
    def expand(features: dict) -> dict:
        """Compacted `features` as LightGlue takes them."""
        return {**features, "descriptors": features["descriptors"].float()}

    @torch.inference_mode()
    def match(self, a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
        """Matched keypoints in `a` and in `b`, as two `(n, 2)` arrays."""
        from lightglue.utils import rbd

        none = np.zeros((0, 2), dtype=np.float32)
        if a["keypoints"].shape[1] < 2 or b["keypoints"].shape[1] < 2:
            return none, none
        found = rbd(self.matcher({"image0": a, "image1": b}))["matches"]
        pts_a = a["keypoints"][0][found[:, 0]]
        pts_b = b["keypoints"][0][found[:, 1]]
        return pts_a.cpu().numpy(), pts_b.cpu().numpy()


def inliers(
    pts_a: np.ndarray, pts_b: np.ndarray, settings: Settings, fundamental: bool
) -> np.ndarray:
    if len(pts_a) < 8:
        return np.zeros(len(pts_a), dtype=bool)
    if fundamental:
        _, mask = cv2.findFundamentalMat(
            pts_a, pts_b, cv2.USAC_MAGSAC, settings.f_px, 0.999, 10_000
        )
    else:
        _, mask = cv2.findHomography(
            pts_a,
            pts_b,
            cv2.USAC_MAGSAC,
            settings.h_px,
            maxIters=10_000,
            confidence=0.999,
        )
    if mask is None:
        return np.zeros(len(pts_a), dtype=bool)
    return mask.ravel().astype(bool)


def cells(points: np.ndarray, shape: tuple[int, ...], grid: int) -> int:
    height, width = shape[:2]
    x = np.minimum((points[:, 0] / width * grid).astype(int), grid - 1)
    y = np.minimum((points[:, 1] / height * grid).astype(int), grid - 1)
    return len(np.unique(y * grid + x))


def geometry(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    settings: Settings,
) -> dict:
    """The scores of one matched pair. `cells` is over the fundamental matrix
    inliers, the lower of the two images."""
    f = inliers(pts_a, pts_b, settings, fundamental=True)
    h = inliers(pts_a, pts_b, settings, fundamental=False)
    return {
        "matches": len(pts_a),
        "f_inliers": int(f.sum()),
        "h_inliers": int(h.sum()),
        "cells": min(
            cells(pts_a[f], shape_a, settings.grid),
            cells(pts_b[f], shape_b, settings.grid),
        ),
    }
