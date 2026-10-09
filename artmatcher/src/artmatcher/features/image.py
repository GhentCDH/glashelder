"""Image features of a pair: ORB geometric verification and perceptual hashes,
plus the aligned-overlap comparison the homography makes possible."""

import io
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import cv2
import imagehash
import numpy as np
from PIL import Image

from common.images import is_url
from common.images import load as load_pil

from ..stitching import find_homography


def make_orb() -> tuple:
    """The ORB detector and the matcher for its descriptors, as `(orb, bf)`."""
    orb = cv2.ORB_create(nfeatures=1000)  # ty: ignore[unresolved-attribute]
    return orb, cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


MIN_ALIGN_INLIERS = 15
"""ORB inliers a homography needs before the pair is aligned and compared."""
MIN_ALIGN_SIDE = 32
"""Smallest side, in reference pixels, of an overlap worth comparing."""
ALIGN_SIZE = 256
"""Longest side of the aligned crops."""
ALIGN_MARGIN = 0.02
"""Part of the overlap cut from each side, so warp borders stay out of the crops."""


@dataclass(frozen=True, slots=True)
class HashDistances:
    """Perceptual hash distances of a pair; the defaults mean no similarity."""

    phash: float = 64.0
    dhash: float = 64.0
    whash: float = 64.0
    ahash: float = 64.0

    @classmethod
    def between(cls, a: Image.Image, b: Image.Image) -> Self:
        """All hashes work on grayscale internally, so color does not count."""
        try:
            return cls(
                phash=float(imagehash.phash(a) - imagehash.phash(b)),
                dhash=float(imagehash.dhash(a) - imagehash.dhash(b)),
                whash=float(imagehash.whash(a) - imagehash.whash(b)),
                ahash=float(imagehash.average_hash(a) - imagehash.average_hash(b)),
            )
        except Exception:
            return cls()

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}{f.name}_distance": getattr(self, f.name) for f in fields(self)
        }


@dataclass(frozen=True, slots=True)
class OrbFeatures:
    """ORB geometric verification of a pair."""

    num_matches: int = 0
    num_inliers: int = 0
    confidence: float = 0.0
    warp_stretch: float = 1.0

    @property
    def inlier_match_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "orb_num_matches": float(self.num_matches),
            "orb_num_inliers": float(self.num_inliers),
            "orb_confidence": self.confidence,
            "orb_warp_stretch": self.warp_stretch,
            "inlier_match_ratio": self.inlier_match_ratio,
        }


@dataclass(frozen=True, slots=True)
class AlignedFeatures:
    """Comparison of the overlap of a pair after the ORB homography aligned it;
    the defaults are for pairs that could not be aligned."""

    hashes: HashDistances = HashDistances()
    ncc: float = 0.0
    ssim: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.hashes.as_dict("aligned_") | {
            "aligned_ncc": self.ncc,
            "aligned_ssim": self.ssim,
        }


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    """Everything the classifier computes from the images of a pair."""

    orb: OrbFeatures = OrbFeatures()
    hashes: HashDistances = HashDistances()
    aligned: AlignedFeatures = AlignedFeatures()

    def as_dict(self) -> dict[str, float]:
        return self.orb.as_dict() | self.hashes.as_dict() | self.aligned.as_dict()


NO_IMAGE_FEATURES = ImageFeatures().as_dict()
"""Image features of a pair whose images could not be compared."""


def _load_views(ref: str) -> tuple[np.ndarray, Image.Image]:
    """One image as both views the features need: BGR for ORB, RGB for the hashes.

    Read once, since a reference is a download and a glashelder image a read from
    a network share. Each view is decoded exactly as when they were loaded
    separately (OpenCV for files, Pillow for URLs and for the hashes), so the
    features stay the ones the model was trained on.
    """
    if is_url(ref):
        image = load_pil(ref)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), image
    raw = Path(ref).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not load image {ref}")
    return bgr, Image.open(io.BytesIO(raw)).convert("RGB")


def pair_features(query_path: str, reference_path: str, orb, bf) -> ImageFeatures:
    """Image features of a pair, reading each image only once."""
    try:
        query_bgr, query_rgb = _load_views(query_path)
        reference_bgr, reference_rgb = _load_views(reference_path)
    except Exception:
        return ImageFeatures()
    query_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    orb_feats, h = _orb_features(query_gray, ref_gray, orb, bf)
    return ImageFeatures(
        orb=orb_feats,
        hashes=HashDistances.between(query_rgb, reference_rgb),
        aligned=_aligned_features(query_gray, ref_gray, h, orb_feats.num_inliers),
    )


def _orb_matches(query_gray: np.ndarray, ref_gray: np.ndarray, orb, bf) -> tuple:
    """ORB keypoints of two grayscale images and the matches that pass the ratio test.

    Returns `(query_kp, ref_kp, good_matches)`; no matches when either image has
    too few keypoints to match.
    """
    query_kp, query_des = orb.detectAndCompute(query_gray, None)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
    if query_des is None or ref_des is None or len(query_kp) == 0:
        return query_kp, ref_kp, []
    if len(query_des) < 2 or len(ref_des) < 2:
        return query_kp, ref_kp, []

    # Use knnMatch with ratio test for ORB (like SIFT)
    good_matches = []
    for match in bf.knnMatch(query_des, ref_des, k=2):
        if len(match) == 2:
            m, n = match
            # Ratio test (slightly higher threshold for binary descriptors)
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    return query_kp, ref_kp, good_matches


def orb_overlay(
    query_path: str, reference_path: str, max_side: int = 1200
) -> np.ndarray | None:
    """The query image warped onto the reference with the ORB homography, blended
    half and half over it, as BGR; None when ORB finds no homography.

    The same matching as the ORB features, so this shows what they measured.
    """
    query_img, _ = _load_views(query_path)
    ref_img, _ = _load_views(reference_path)
    orb, bf = make_orb()
    query_kp, ref_kp, good_matches = _orb_matches(
        cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY),
        orb,
        bf,
    )
    h, _ = find_homography(query_kp, ref_kp, good_matches)
    if h is None:
        return None

    height, width = ref_img.shape[:2]
    warped = cv2.warpPerspective(query_img, h, (width, height))
    covered = cv2.warpPerspective(
        np.full(query_img.shape[:2], 255, np.uint8), h, (width, height)
    )
    blended = cv2.addWeighted(ref_img, 0.5, warped, 0.5, 0)
    # outside the warped image there is only the reference to show
    out = np.where(covered[..., None] > 0, blended, ref_img)
    scale = max_side / max(height, width)
    if scale < 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


def _query_corners(query_gray: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Corners of the query (TL, TR, BR, BL) in reference coordinates, as (4, 2)."""
    img_h, img_w = query_gray.shape[:2]
    corners = np.array(
        [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
    )
    return cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h).reshape(-1, 2)


def _warp_stretch(query_gray: np.ndarray, corners: np.ndarray) -> float:
    """Largest factor by which the homography stretches or shrinks a query edge."""
    img_h, img_w = query_gray.shape[:2]
    warped_edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    with np.errstate(divide="ignore"):
        ratios = warped_edges / np.array([img_w, img_h, img_w, img_h])
        return float(np.maximum(ratios, 1 / ratios).max())


def _orb_features(
    query_gray: np.ndarray, ref_gray: np.ndarray, orb, bf
) -> tuple[OrbFeatures, np.ndarray | None]:
    """ORB geometric verification of two grayscale images (much faster than SIFT),
    with the homography from the query onto the reference when there is one."""
    try:
        query_kp, ref_kp, good_matches = _orb_matches(query_gray, ref_gray, orb, bf)
        h, num_inliers = find_homography(query_kp, ref_kp, good_matches)
        if h is None:
            return OrbFeatures(num_matches=len(good_matches)), None
        orb_feats = OrbFeatures(
            num_matches=len(good_matches),
            num_inliers=num_inliers,
            confidence=num_inliers / len(query_kp),
            warp_stretch=_warp_stretch(query_gray, _query_corners(query_gray, h)),
        )
        return orb_feats, h
    except Exception:
        return OrbFeatures(), None


def _aligned_pair(
    query_gray: np.ndarray, ref_gray: np.ndarray, h: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """The overlap of a pair as two grayscale crops of the same size, the query
    warped onto the reference with `h`; None when the warp folds the query or
    the overlap is too small.

    The overlap is the axis-aligned rectangle inside the warped query, so no crop
    holds pixels from outside its image (the hashes cannot mask them out).
    """
    corners = _query_corners(query_gray, h)
    if not cv2.isContourConvex(corners):
        return None
    tl, tr, br, bl = corners
    ref_h, ref_w = ref_gray.shape[:2]
    x0, x1 = max(tl[0], bl[0], 0.0), min(tr[0], br[0], float(ref_w))
    y0, y1 = max(tl[1], tr[1], 0.0), min(bl[1], br[1], float(ref_h))
    margin_x, margin_y = (x1 - x0) * ALIGN_MARGIN, (y1 - y0) * ALIGN_MARGIN
    left, top = math.ceil(x0 + margin_x), math.ceil(y0 + margin_y)
    right, bottom = math.floor(x1 - margin_x), math.floor(y1 - margin_y)
    width, height = right - left, bottom - top
    if min(width, height) < MIN_ALIGN_SIDE:
        return None

    scale = min(1.0, ALIGN_SIZE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    ref_crop = cv2.resize(
        ref_gray[top:bottom, left:right], size, interpolation=cv2.INTER_AREA
    )

    # reference coordinates to crop coordinates, the same mapping as the resize
    scale_x, scale_y = size[0] / width, size[1] / height
    to_crop = np.array(
        [[scale_x, 0, -left * scale_x], [0, scale_y, -top * scale_y], [0, 0, 1]]
    )
    # warpPerspective aliases when it shrinks a lot, so a query much larger than
    # the crop is first shrunk to about the crop's resolution
    query_h, query_w = query_gray.shape[:2]
    shrink = math.sqrt(
        cv2.contourArea(corners) * scale_x * scale_y / (query_w * query_h)
    )
    matrix = to_crop @ h
    if shrink < 1:
        query_gray = cv2.resize(
            query_gray,
            (max(1, round(query_w * shrink)), max(1, round(query_h * shrink))),
            interpolation=cv2.INTER_AREA,
        )
        unshrink = np.diag(
            [query_w / query_gray.shape[1], query_h / query_gray.shape[0], 1.0]
        )
        matrix = matrix @ unshrink
    query_crop = cv2.warpPerspective(
        query_gray,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return query_crop, ref_crop


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two same-sized images, in [-1, 1]."""
    da = a - a.mean(dtype=np.float64)
    db = b - b.mean(dtype=np.float64)
    denom = math.sqrt(float((da * da).sum() * (db * db).sum()))
    return float((da * db).sum()) / denom if denom > 0 else 0.0


def _blur(image: np.ndarray) -> np.ndarray:
    """The Gaussian window of SSIM."""
    return cv2.GaussianBlur(image, (11, 11), 1.5)


def _ssim(query: np.ndarray, ref: np.ndarray) -> float:
    """Mean SSIM of two same-sized grayscale images.

    The query first gets the mean and spread of the reference, so a black and
    white plate of a colour painting is judged on structure, not on brightness
    or contrast.
    """
    x = ref.astype(np.float64)
    y = query.astype(np.float64)
    spread = y.std()
    if spread > 0:
        y = (y - y.mean()) * (x.std() / spread) + x.mean()

    mu_x, mu_y = _blur(x), _blur(y)
    var_x = _blur(x * x) - mu_x * mu_x
    var_y = _blur(y * y) - mu_y * mu_y
    cov = _blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return float(ssim.mean())


def _aligned_features(
    query_gray: np.ndarray,
    ref_gray: np.ndarray,
    h: np.ndarray | None,
    num_inliers: int,
) -> AlignedFeatures:
    """Hashes, NCC and SSIM of the overlap of a pair, when a good homography
    aligns it."""
    if h is None or num_inliers < MIN_ALIGN_INLIERS:
        return AlignedFeatures()
    try:
        pair = _aligned_pair(query_gray, ref_gray, h)
    except cv2.error:
        return AlignedFeatures()
    if pair is None:
        return AlignedFeatures()
    query_crop, ref_crop = pair
    return AlignedFeatures(
        hashes=HashDistances.between(
            Image.fromarray(query_crop), Image.fromarray(ref_crop)
        ),
        ncc=_ncc(query_crop, ref_crop),
        ssim=_ssim(query_crop, ref_crop),
    )
