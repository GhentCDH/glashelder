"""ML-based classifier for artwork matching using Random Forest."""

import io
import json
import logging
import pickle
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import cv2
import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from common.images import is_url
from common.images import load as load_pil

from .stitching import compute_homography

logger = logging.getLogger(__name__)
console = Console()


# Global variables for worker processes
_worker_orb = None
_worker_bf = None


def _init_orb_worker() -> None:
    """Initialize worker process with ORB detector and BFMatcher."""
    global _worker_orb, _worker_bf
    _worker_orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
    _worker_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _extract_orb_worker(
    args: tuple[int, str, str, list[float], int], timeout=25
) -> tuple[int, dict[str, float], int]:
    """Worker function to extract features for a single entry.

    Args:
        args: Tuple of (index, glashelder_path, wikiart_path, similarities, label)

    Returns:
        Tuple of (index, features_dict, label)
    """
    global _worker_orb, _worker_bf

    def handler(signum, frame):
        raise TimeoutError("Worker timeout")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        idx, glashelder_path, wikiart_path, similarities, label = args

        # Initialize if needed (shouldn't happen with initializer, but be safe)
        if _worker_orb is None:
            _worker_orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
        if _worker_bf is None:
            _worker_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Extract distribution features first (fast)
        features = _extract_distribution_features(similarities)

        orb_feats, hash_feats = _pair_features(
            glashelder_path, wikiart_path, _worker_orb, _worker_bf
        )
        features["orb_num_matches"] = float(orb_feats["num_matches"])
        features["orb_num_inliers"] = float(orb_feats["num_inliers"])
        features["orb_confidence"] = orb_feats["confidence"]
        features["orb_warp_stretch"] = orb_feats["warp_stretch"]
        features["inlier_match_ratio"] = (
            orb_feats["num_inliers"] / orb_feats["num_matches"]
            if orb_feats["num_matches"] > 0
            else 0.0
        )

        features["phash_distance"] = hash_feats["phash_distance"]
        features["dhash_distance"] = hash_feats["dhash_distance"]
        features["whash_distance"] = hash_feats["whash_distance"]
        features["ahash_distance"] = hash_feats["ahash_distance"]

        # Sanitize to remove NaN/Inf values
        features = _sanitize_features(features)

        return (idx, features, label)

    except TimeoutError:
        idx, glashelder_path, _, similarities, label = args
        logger.warning(f"Worker timeout for {glashelder_path}")
        features = _extract_distribution_features(similarities)
        # Fill in defaults for image-based features
        features.update(
            {
                "orb_num_matches": 0.0,
                "orb_num_inliers": 0.0,
                "orb_confidence": 0.0,
                "orb_warp_stretch": 1.0,
                "inlier_match_ratio": 0.0,
                "phash_distance": 64.0,
                "dhash_distance": 64.0,
                "whash_distance": 64.0,
                "ahash_distance": 64.0,
            }
        )
        return (idx, _sanitize_features(features), label)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _sanitize_features(features: dict[str, float]) -> dict[str, float]:
    """Replace NaN and Inf values with safe defaults."""
    sanitized = {}
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            # Use 0 as default for most features, 1.0 for ratios and warp_stretch
            if "ratio" in key or "warp_stretch" in key:
                sanitized[key] = 1.0
            elif "hash_distance" in key:
                sanitized[key] = 64.0  # Max hamming distance for 64-bit hash
            else:
                sanitized[key] = 0.0
        else:
            # Clip to reasonable range to avoid sklearn issues
            sanitized[key] = float(np.clip(value, -1e6, 1e6))
    return sanitized


def _extract_distribution_features(similarities: list[float]) -> dict[str, float]:
    """Extract distribution shape features from similarity scores (standalone function)."""
    features = {}
    sims = np.array(similarities)
    n = len(sims)

    features["sim_0"] = float(sims[0]) if n > 0 else 0.0
    features["sim_1"] = float(sims[1]) if n > 1 else 0.0
    features["sim_2"] = float(sims[2]) if n > 2 else 0.0

    features["gap_01"] = features["sim_0"] - features["sim_1"]
    features["gap_02"] = features["sim_0"] - features["sim_2"]
    features["gap_12"] = features["sim_1"] - features["sim_2"]

    rest = sims[1:10] if n > 1 else np.array([0.0])
    features["mean_1to9"] = float(np.mean(rest))
    features["std_1to9"] = float(np.std(rest)) if len(rest) > 1 else 0.0

    # Z-score with minimum std threshold to avoid huge values
    std_threshold = 1e-6
    if features["std_1to9"] > std_threshold:
        features["sim_0_zscore"] = (
            features["sim_0"] - features["mean_1to9"]
        ) / features["std_1to9"]
    else:
        features["sim_0_zscore"] = 0.0

    if len(rest) > 1:
        x = np.arange(len(rest))
        try:
            coeffs = np.polyfit(x, rest, 1)
            features["decay_rate"] = float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            features["decay_rate"] = 0.0
    else:
        features["decay_rate"] = 0.0

    features["sim_min"] = float(np.min(sims[:10])) if n > 0 else 0.0
    features["sim_range"] = features["sim_0"] - features["sim_min"]

    # Poly diff
    if len(similarities) >= 2:
        rest_list = similarities[1:]
        n_rest = len(rest_list)
        if n_rest == 1:
            weights = np.array([1.0])
        else:
            max_distance = n_rest - 1
            sigma = max_distance / np.sqrt(-2 * np.log(0.3))
            indices = np.arange(n_rest)
            weights = np.exp(-0.5 * (indices / sigma) ** 2)
            weights = weights / weights.sum()
        x = np.arange(n_rest)
        try:
            coeffs = np.polyfit(x, rest_list, 1, w=weights)
            features["poly_diff"] = similarities[0] - coeffs[-1]
        except (np.linalg.LinAlgError, ValueError):
            features["poly_diff"] = 0.0
    else:
        features["poly_diff"] = 0.0

    # Ratio with minimum denominator threshold
    min_denom = 1e-6
    features["sim_ratio_01"] = (
        features["sim_0"] / max(features["sim_1"], min_denom)
        if features["sim_1"] > min_denom
        else 1.0
    )

    return features


NO_ORB: dict[str, float] = {
    "num_matches": 0,
    "num_inliers": 0,
    "confidence": 0.0,
    "warp_stretch": 1.0,
}
NO_HASH: dict[str, float] = {
    "phash_distance": 64.0,  # max distance (no similarity)
    "dhash_distance": 64.0,
    "whash_distance": 64.0,
    "ahash_distance": 64.0,
}


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


def _pair_features(
    query_path: str, reference_path: str, orb, bf
) -> tuple[dict[str, float], dict[str, float]]:
    """ORB and hash features of a pair, reading each image only once."""
    try:
        query_bgr, query_rgb = _load_views(query_path)
        reference_bgr, reference_rgb = _load_views(reference_path)
    except Exception:
        return dict(NO_ORB), dict(NO_HASH)
    return (
        _orb_features(query_bgr, reference_bgr, orb, bf),
        _hash_features(query_rgb, reference_rgb),
    )


def _orb_matches(query_img: np.ndarray, ref_img: np.ndarray, orb, bf) -> tuple:
    """ORB keypoints of two BGR images and the matches that pass the ratio test.

    Returns `(query_kp, ref_kp, good_matches)`; no matches when either image has
    too few keypoints to match.
    """
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

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
    orb = cv2.ORB_create(nfeatures=1000)  # ty: ignore[unresolved-attribute]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    query_kp, ref_kp, good_matches = _orb_matches(query_img, ref_img, orb, bf)
    h = compute_homography(query_kp, ref_kp, good_matches)
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


def _orb_features(
    query_img: np.ndarray, ref_img: np.ndarray, orb, bf
) -> dict[str, float]:
    """ORB geometric verification of two BGR images (much faster than SIFT)."""
    try:
        query_kp, ref_kp, good_matches = _orb_matches(query_img, ref_img, orb, bf)
        num_matches = len(good_matches)

        if num_matches < 4:
            return {
                "num_matches": num_matches,
                "num_inliers": 0,
                "confidence": 0.0,
                "warp_stretch": 1.0,
            }

        h = compute_homography(query_kp, ref_kp, good_matches)

        if h is None:
            return {
                "num_matches": num_matches,
                "num_inliers": 0,
                "confidence": 0.0,
                "warp_stretch": 1.0,
            }

        src_pts = np.array(
            [query_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        dst_pts = np.array(
            [ref_kp[m.trainIdx].pt for m in good_matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        num_inliers = int(mask.ravel().sum()) if mask is not None else 0

        confidence = num_inliers / len(query_kp) if len(query_kp) > 0 else 0.0

        # Compute warp stretch
        img_h, img_w = query_img.shape[:2]
        corners = np.array(
            [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
        ).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
            orig_edges = [img_w, img_h, img_w, img_h]
            warped_edges = []
            for i in range(4):
                p1 = warped[i]
                p2 = warped[(i + 1) % 4]
                warped_edges.append(np.linalg.norm(p2 - p1))
            ratios = []
            for orig, warped_len in zip(orig_edges, warped_edges, strict=True):
                if orig > 0:
                    ratios.append(max(warped_len / orig, orig / warped_len))
            warp_stretch = max(ratios) if ratios else 1.0
        except cv2.error:
            warp_stretch = 1.0

        return {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "confidence": confidence,
            "warp_stretch": warp_stretch,
        }

    except Exception:
        return {
            "num_matches": 0,
            "num_inliers": 0,
            "confidence": 0.0,
            "warp_stretch": 1.0,
        }


def _hash_features(query_img: Image.Image, ref_img: Image.Image) -> dict[str, float]:
    """Perceptual hash distances of two RGB images (very fast).

    Uses multiple hash types to capture different image properties.
    All hashes work on grayscale internally, so color differences don't affect results.
    """
    try:
        # Perceptual hash (DCT-based, good for similar images)
        q_phash = imagehash.phash(query_img)
        r_phash = imagehash.phash(ref_img)

        # Difference hash (gradient-based, captures structure)
        q_dhash = imagehash.dhash(query_img)
        r_dhash = imagehash.dhash(ref_img)

        # Wavelet hash (multi-resolution, good for scaled images)
        q_whash = imagehash.whash(query_img)
        r_whash = imagehash.whash(ref_img)

        # Average hash (simple but fast)
        q_ahash = imagehash.average_hash(query_img)
        r_ahash = imagehash.average_hash(ref_img)

        return {
            "phash_distance": float(q_phash - r_phash),
            "dhash_distance": float(q_dhash - r_dhash),
            "whash_distance": float(q_whash - r_whash),
            "ahash_distance": float(q_ahash - r_ahash),
        }

    except Exception:
        return {
            "phash_distance": 64.0,  # Max distance (no similarity)
            "dhash_distance": 64.0,
            "whash_distance": 64.0,
            "ahash_distance": 64.0,
        }


@dataclass
class ORBFeatures:
    """ORB geometric verification features for a single image pair."""

    num_matches: int
    num_inliers: int
    confidence: float
    warp_stretch: float

    @classmethod
    def empty(cls) -> "ORBFeatures":
        """Return empty ORB features (when extraction fails)."""
        return cls(num_matches=0, num_inliers=0, confidence=0.0, warp_stretch=1.0)


@dataclass
class HashFeatures:
    """Perceptual hash features for a single image pair."""

    phash_distance: float
    dhash_distance: float
    whash_distance: float
    ahash_distance: float

    @classmethod
    def empty(cls) -> "HashFeatures":
        """Return empty hash features (max distance = no similarity)."""
        return cls(
            phash_distance=64.0,
            dhash_distance=64.0,
            whash_distance=64.0,
            ahash_distance=64.0,
        )


@dataclass
class MLClassificationResult:
    """Result of ML classification containing the decision and confidence."""

    is_match: bool
    confidence: float
    threshold: float
    features: dict[str, float] = field(default_factory=dict)


class FeatureExtractor:
    """Extract features for ML classification from similarities and image paths."""

    def __init__(self):
        self._orb = None
        self._bf = None

    def _get_orb(self):
        """Lazy initialization of ORB detector."""
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
        return self._orb

    def _get_bf(self):
        """Lazy initialization of BFMatcher for ORB."""
        if self._bf is None:
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return self._bf

    def extract_pair_features(
        self, query_path: str, reference_path: str
    ) -> tuple[ORBFeatures, HashFeatures]:
        """ORB and hash features between two images, reading each image once."""
        found, hashes = _pair_features(
            query_path, reference_path, self._get_orb(), self._get_bf()
        )
        # the worker speaks the flat float dict the feature vector wants
        orb = ORBFeatures(
            num_matches=int(found["num_matches"]),
            num_inliers=int(found["num_inliers"]),
            confidence=found["confidence"],
            warp_stretch=found["warp_stretch"],
        )
        return orb, HashFeatures(**hashes)

    def extract_orb_features(self, query_path: str, reference_path: str) -> ORBFeatures:
        """Extract ORB geometric verification features between two images."""
        return self.extract_pair_features(query_path, reference_path)[0]

    def extract_hash_features(
        self, query_path: str, reference_path: str
    ) -> HashFeatures:
        """Extract perceptual hash features between two images."""
        return self.extract_pair_features(query_path, reference_path)[1]

    def _compute_warp_stretch(self, h: NDArray, img_shape: tuple[int, int]) -> float:
        """Compute how much a homography stretches/distorts an image.

        Returns the maximum ratio of (warped edge length / original edge length).
        A value of 1.0 means no stretch, higher values mean more distortion.
        """
        img_h, img_w = img_shape[:2]
        corners = np.array(
            [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
        ).reshape(-1, 1, 2)

        try:
            warped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
        except cv2.error:
            return 1.0

        # Original edge lengths
        orig_edges = [img_w, img_h, img_w, img_h]

        # Warped edge lengths
        warped_edges = []
        for i in range(4):
            p1 = warped[i]
            p2 = warped[(i + 1) % 4]
            warped_edges.append(np.linalg.norm(p2 - p1))

        # Compute stretch ratios
        ratios = []
        for orig, warped_len in zip(orig_edges, warped_edges, strict=True):
            if orig > 0:
                ratios.append(max(warped_len / orig, orig / warped_len))

        return max(ratios) if ratios else 1.0

    def extract_distribution_features(
        self, similarities: list[float]
    ) -> dict[str, float]:
        """Extract distribution shape features from similarity scores.

        Args:
            similarities: List of top-k similarity scores (descending order)

        Returns:
            Dictionary of distribution features
        """
        features = {}

        # Ensure we have enough similarities
        sims = np.array(similarities)
        n = len(sims)

        # Top similarities
        features["sim_0"] = float(sims[0]) if n > 0 else 0.0
        features["sim_1"] = float(sims[1]) if n > 1 else 0.0
        features["sim_2"] = float(sims[2]) if n > 2 else 0.0

        # Gaps between top similarities (key discriminators)
        features["gap_01"] = features["sim_0"] - features["sim_1"]
        features["gap_02"] = features["sim_0"] - features["sim_2"]
        features["gap_12"] = features["sim_1"] - features["sim_2"]

        # Distribution shape features (using similarities[1:10])
        rest = sims[1:10] if n > 1 else np.array([0.0])
        features["mean_1to9"] = float(np.mean(rest))
        features["std_1to9"] = float(np.std(rest)) if len(rest) > 1 else 0.0

        # Z-score of top similarity with minimum std threshold
        std_threshold = 1e-6
        if features["std_1to9"] > std_threshold:
            features["sim_0_zscore"] = (
                features["sim_0"] - features["mean_1to9"]
            ) / features["std_1to9"]
        else:
            features["sim_0_zscore"] = 0.0

        # Decay rate (linear slope of similarities[1:10])
        if len(rest) > 1:
            x = np.arange(len(rest))
            try:
                coeffs = np.polyfit(x, rest, 1)
                features["decay_rate"] = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                features["decay_rate"] = 0.0
        else:
            features["decay_rate"] = 0.0

        # Min and range
        features["sim_min"] = float(np.min(sims[:10])) if n > 0 else 0.0
        features["sim_range"] = features["sim_0"] - features["sim_min"]

        # Poly diff (SimpleClassifier's diff value)
        features["poly_diff"] = self._compute_poly_diff(similarities)

        # Ratio features with minimum denominator threshold
        min_denom = 1e-6
        features["sim_ratio_01"] = (
            features["sim_0"] / max(features["sim_1"], min_denom)
            if features["sim_1"] > min_denom
            else 1.0
        )

        return features

    def _compute_poly_diff(self, similarities: list[float]) -> float:
        """Compute the polynomial diff value used by SimpleClassifier."""
        if len(similarities) < 2:
            return 0.0

        rest = similarities[1:]
        n = len(rest)

        # Gaussian weights with minimum value 0.3
        if n == 1:
            weights = np.array([1.0])
        else:
            max_distance = n - 1
            sigma = max_distance / np.sqrt(-2 * np.log(0.3))
            indices = np.arange(n)
            weights = np.exp(-0.5 * (indices / sigma) ** 2)
            weights = weights / weights.sum()

        # Fit weighted linear polynomial
        x = np.arange(n)
        try:
            coeffs = np.polyfit(x, rest, 1, w=weights)
            at_zero = coeffs[-1]  # Intercept
            return similarities[0] - at_zero
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def extract_all_features(
        self,
        similarities: list[float],
        glashelder_path: str | None = None,
        wikiart_path: str | None = None,
    ) -> dict[str, float]:
        """Extract all features for ML classification.

        Args:
            similarities: List of top-k similarity scores
            glashelder_path: Path to glashelder image (optional, for ORB/hash)
            wikiart_path: Path to wikiart image (optional, for ORB/hash)

        Returns:
            Dictionary with all features (distribution + ORB + hash)
        """
        features = self.extract_distribution_features(similarities)

        # ORB and hash features (if image paths provided)
        if glashelder_path and wikiart_path:
            orb_feats, hash_feats = self.extract_pair_features(
                glashelder_path, wikiart_path
            )
            features["orb_num_matches"] = float(orb_feats.num_matches)
            features["orb_num_inliers"] = float(orb_feats.num_inliers)
            features["orb_confidence"] = orb_feats.confidence
            features["orb_warp_stretch"] = orb_feats.warp_stretch

            # Inlier/match ratio
            features["inlier_match_ratio"] = (
                orb_feats.num_inliers / orb_feats.num_matches
                if orb_feats.num_matches > 0
                else 0.0
            )

            features["phash_distance"] = hash_feats.phash_distance
            features["dhash_distance"] = hash_feats.dhash_distance
            features["whash_distance"] = hash_feats.whash_distance
            features["ahash_distance"] = hash_feats.ahash_distance
        else:
            # Placeholder features when paths not available
            features["orb_num_matches"] = 0.0
            features["orb_num_inliers"] = 0.0
            features["orb_confidence"] = 0.0
            features["orb_warp_stretch"] = 1.0
            features["inlier_match_ratio"] = 0.0
            features["phash_distance"] = 64.0
            features["dhash_distance"] = 64.0
            features["whash_distance"] = 64.0
            features["ahash_distance"] = 64.0

        # Sanitize to remove NaN/Inf values
        return _sanitize_features(features)


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    cv_accuracy: float
    cv_std: float
    feature_importance: dict[str, float]
    n_samples: int
    n_positives: int
    n_negatives: int


class MLClassifier:
    """ML-based classifier using Random Forest with calibrated probabilities."""

    FEATURE_ORDER: ClassVar[list[str]] = [
        "sim_0",
        "sim_1",
        "sim_2",
        "gap_01",
        "gap_02",
        "gap_12",
        "orb_num_matches",
        "orb_num_inliers",
        "orb_confidence",
        "orb_warp_stretch",
        "mean_1to9",
        "std_1to9",
        "sim_0_zscore",
        "decay_rate",
        "sim_min",
        "sim_range",
        "poly_diff",
        "sim_ratio_01",
        "inlier_match_ratio",
        "phash_distance",
        "dhash_distance",
        "whash_distance",
        "ahash_distance",
    ]

    def __init__(self, threshold: float = 0.5):
        """Initialize the classifier.

        Args:
            threshold: Confidence threshold for is_match decision
        """
        self.threshold = threshold
        self._model = None
        self._feature_extractor = FeatureExtractor()

    def _features_to_array(self, features: dict[str, float]) -> NDArray:
        """Convert feature dict to ordered numpy array."""
        return np.array([features.get(f, 0.0) for f in self.FEATURE_ORDER])

    def train(
        self,
        dataset_path: str,
        n_estimators: int = 100,
        cv_folds: int = 5,
        extract_image_features: bool = True,
        workers: int = 1,
    ) -> TrainingMetrics:
        """Train the classifier from a labeled JSON dataset.

        Args:
            dataset_path: Path to labeling_dataset.json from the web UI
            n_estimators: Number of trees in the Random Forest
            cv_folds: Number of cross-validation folds
            extract_image_features: Whether to extract ORB/hash features from image paths
            workers: Number of parallel workers for feature extraction (1=sequential)

        Returns:
            TrainingMetrics with CV accuracy and feature importance
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Load dataset
        console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
        with open(dataset_path) as f:
            dataset = json.load(f)

        # Collect all entries with their labels
        entries_with_labels: list[tuple[dict, int]] = []

        # True positives (classifier correct when predicting match)
        for entry in dataset.get("positives", []):
            entries_with_labels.append((entry, 1))

        # True negatives (classifier correct when predicting no match)
        for entry in dataset.get("negatives", []):
            entries_with_labels.append((entry, 0))

        # False positives (classifier said match but was wrong) -> actual negative
        for entry in dataset.get("false_positives", []):
            entries_with_labels.append((entry, 0))

        # False negatives (classifier said no match but was wrong) -> actual positive
        for entry in dataset.get("false_negatives", []):
            entries_with_labels.append((entry, 1))

        n_entries = len(entries_with_labels)
        console.print(f"[cyan]Found {n_entries} labeled samples[/cyan]")

        # Extract features (parallel or sequential)
        if extract_image_features and workers > 1:
            features_list, labels_list = self._extract_features_parallel(
                entries_with_labels, workers
            )
        else:
            features_list, labels_list = self._extract_features_sequential(
                entries_with_labels, extract_image_features
            )

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        n_samples = len(labels_array)
        n_positives = int(np.sum(labels_array))
        n_negatives = n_samples - n_positives

        console.print(
            f"[green]Training on {n_samples} samples "
            f"({n_positives} positives, {n_negatives} negatives)[/green]"
        )

        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples for training, got {n_samples}")

        # Train Random Forest with cross-validation
        console.print("[cyan]Training Random Forest classifier...[/cyan]")
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validation with progress
        console.print(f"[cyan]Running {cv_folds}-fold cross-validation...[/cyan]")
        n_cv_folds = min(cv_folds, n_samples)
        cv_scores = cross_val_score(
            base_rf, features_array, labels_array, cv=n_cv_folds, scoring="accuracy"
        )
        cv_accuracy = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        console.print(
            f"[green]Cross-validation accuracy: {cv_accuracy:.3f} (+/- {cv_std:.3f})[/green]"
        )

        # Train final model with probability calibration
        console.print("[cyan]Calibrating probabilities...[/cyan]")
        self._model = CalibratedClassifierCV(
            base_rf,
            method="sigmoid",
            cv=min(3, n_samples),
        )
        self._model.fit(features_array, labels_array)

        # Get feature importance from the base estimator
        console.print("[cyan]Computing feature importance...[/cyan]")
        base_rf.fit(features_array, labels_array)
        importances = base_rf.feature_importances_
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_ORDER, importances, strict=True)
        }

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        console.print("\n[bold green]Top 5 features by importance:[/bold green]")
        for i, (name, imp) in enumerate(list(feature_importance.items())[:5]):
            console.print(f"  {i + 1}. [yellow]{name}[/yellow]: {imp:.4f}")

        return TrainingMetrics(
            cv_accuracy=cv_accuracy,
            cv_std=cv_std,
            feature_importance=feature_importance,
            n_samples=n_samples,
            n_positives=n_positives,
            n_negatives=n_negatives,
        )

    def _extract_features_sequential(
        self,
        entries_with_labels: list[tuple[dict, int]],
        extract_image_features: bool,
    ) -> tuple[list[NDArray], list[int]]:
        """Extract features sequentially with progress bar."""
        features_list = []
        labels_list = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Extracting features...",
                total=len(entries_with_labels),
            )

            for entry, label in entries_with_labels:
                features = self._extract_entry_features(entry, extract_image_features)
                features_list.append(self._features_to_array(features))
                labels_list.append(label)
                progress.advance(task)

        return features_list, labels_list

    def _extract_features_parallel(
        self,
        entries_with_labels: list[tuple[dict, int]],
        workers: int,
    ) -> tuple[list[NDArray], list[int]]:
        """Extract features in parallel using multiprocessing with progress bar."""
        # Prepare work items: (index, glashelder_path, wikiart_path, similarities, label)
        work_items = []
        for idx, (entry, label) in enumerate(entries_with_labels):
            work_items.append(
                (
                    idx,
                    entry.get("glashelder_path", ""),
                    entry.get("wikiart_path", ""),
                    entry.get("similarities", []),
                    label,
                )
            )

        n_items = len(work_items)
        console.print(
            f"[cyan]Starting {workers} worker processes for ORB/hash extraction...[/cyan]"
        )

        # Results storage (indexed by original position)
        results: dict[int, tuple[dict[str, float], int]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Extracting ORB/hash features...",
                total=n_items,
            )

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_orb_worker,
            ) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(_extract_orb_worker, item): item[0]
                    for item in work_items
                }

                # Process results as they complete
                for future in as_completed(future_to_idx):
                    idx, features, label = future.result()
                    results[idx] = (features, label)
                    progress.advance(task)

        # Reconstruct ordered lists
        features_list = []
        labels_list = []
        for idx in range(n_items):
            features, label = results[idx]
            features_list.append(self._features_to_array(features))
            labels_list.append(label)

        return features_list, labels_list

    def _extract_entry_features(
        self, entry: dict, extract_image_features: bool
    ) -> dict[str, float]:
        """Extract features from a dataset entry."""
        similarities = entry.get("similarities", [])

        glashelder_path = (
            entry.get("glashelder_path") if extract_image_features else None
        )
        wikiart_path = entry.get("wikiart_path") if extract_image_features else None

        return self._feature_extractor.extract_all_features(
            similarities, glashelder_path, wikiart_path
        )

    def classify(
        self,
        similarities: list[float],
        glashelder_path: str | None = None,
        wikiart_path: str | None = None,
    ) -> MLClassificationResult:
        """Classify whether the top match is a true match.

        Args:
            similarities: List of top-k similarity scores
            glashelder_path: Path to glashelder image (for SIFT features)
            wikiart_path: Path to wikiart image (for SIFT features)

        Returns:
            MLClassificationResult with is_match, confidence, and features
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        features = self._feature_extractor.extract_all_features(
            similarities, glashelder_path, wikiart_path
        )
        feature_vector = self._features_to_array(features).reshape(1, -1)

        # Get calibrated probability
        proba = self._model.predict_proba(feature_vector)[0]
        confidence = float(proba[1])  # Probability of class 1 (match)

        is_match = confidence > self.threshold

        return MLClassificationResult(
            is_match=is_match,
            confidence=confidence,
            threshold=self.threshold,
            features=features,
        )

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model (.pkl file)
        """
        if self._model is None:
            raise RuntimeError("No model to save. Call train() first.")

        data = {
            "model": self._model,
            "threshold": self.threshold,
            "feature_order": self.FEATURE_ORDER,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "MLClassifier":
        """Load a trained model from disk.

        Args:
            path: Path to the saved model (.pkl file)
            threshold: Optional threshold override (uses saved threshold if None)

        Returns:
            MLClassifier with loaded model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        classifier = cls(threshold=threshold or data.get("threshold", 0.5))
        classifier._model = data["model"]

        # Verify feature order matches
        saved_order = data.get("feature_order", [])
        if saved_order and saved_order != cls.FEATURE_ORDER:
            logger.warning(
                "Feature order in saved model differs from current. "
                "This may cause issues."
            )

        logger.info(f"Model loaded from {path}")
        return classifier
