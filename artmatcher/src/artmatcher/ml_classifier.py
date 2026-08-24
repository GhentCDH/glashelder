"""ML-based classifier for artwork matching using Random Forest."""

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

from .stitching import (
    compute_homography,
    load_image,
)

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

        # Extract ORB features
        orb_feats = _extract_orb_features_worker(
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

        # Extract perceptual hash features
        hash_feats = _extract_hash_features_worker(glashelder_path, wikiart_path)
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


def _extract_orb_features_worker(
    query_path: str, reference_path: str, orb, bf
) -> dict[str, float]:
    """Extract ORB features in worker process (much faster than SIFT)."""
    try:
        query_img = load_image(query_path)
        ref_img = load_image(reference_path)

        # Convert to grayscale for ORB
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        query_kp, query_des = orb.detectAndCompute(query_gray, None)
        ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

        if query_des is None or ref_des is None or len(query_kp) == 0:
            return {
                "num_matches": 0,
                "num_inliers": 0,
                "confidence": 0.0,
                "warp_stretch": 1.0,
            }

        if len(query_des) < 2 or len(ref_des) < 2:
            return {
                "num_matches": 0,
                "num_inliers": 0,
                "confidence": 0.0,
                "warp_stretch": 1.0,
            }

        # Use knnMatch with ratio test for ORB (like SIFT)
        matches = bf.knnMatch(query_des, ref_des, k=2)

        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                # Ratio test (slightly higher threshold for binary descriptors)
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

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


def _extract_hash_features_worker(
    query_path: str, reference_path: str
) -> dict[str, float]:
    """Extract perceptual hash features (very fast).

    Uses multiple hash types to capture different image properties.
    All hashes work on grayscale internally, so color differences don't affect results.
    """
    try:
        query_img = Image.open(query_path)
        ref_img = Image.open(reference_path)

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

    def extract_orb_features(self, query_path: str, reference_path: str) -> ORBFeatures:
        """Extract ORB geometric verification features between two images.

        Args:
            query_path: Path to the query image (e.g., glashelder photo)
            reference_path: Path to the reference image (e.g., wikiart painting)

        Returns:
            ORBFeatures with geometric verification metrics
        """
        try:
            # Load and extract features from both images
            query_img = load_image(query_path)
            ref_img = load_image(reference_path)

            orb = self._get_orb()
            bf = self._get_bf()

            # Convert to grayscale for ORB
            query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

            query_kp, query_des = orb.detectAndCompute(query_gray, None)
            ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

            if query_des is None or ref_des is None or len(query_kp) == 0:
                return ORBFeatures.empty()

            if len(query_des) < 2 or len(ref_des) < 2:
                return ORBFeatures.empty()

            # Use knnMatch with ratio test
            matches = bf.knnMatch(query_des, ref_des, k=2)

            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            num_matches = len(good_matches)

            if num_matches < 4:
                return ORBFeatures(
                    num_matches=num_matches,
                    num_inliers=0,
                    confidence=0.0,
                    warp_stretch=1.0,
                )

            # Compute homography and count inliers
            h = compute_homography(query_kp, ref_kp, good_matches)

            if h is None:
                return ORBFeatures(
                    num_matches=num_matches,
                    num_inliers=0,
                    confidence=0.0,
                    warp_stretch=1.0,
                )

            # Count inliers using RANSAC mask
            src_pts = np.array(
                [query_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32
            ).reshape(-1, 1, 2)
            dst_pts = np.array(
                [ref_kp[m.trainIdx].pt for m in good_matches], dtype=np.float32
            ).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            num_inliers = int(mask.ravel().sum()) if mask is not None else 0

            # Compute confidence as ratio of inliers to query keypoints
            confidence = num_inliers / len(query_kp) if len(query_kp) > 0 else 0.0

            # Compute warp stretch from homography
            warp_stretch = self._compute_warp_stretch(h, query_img.shape[:2])

            return ORBFeatures(
                num_matches=num_matches,
                num_inliers=num_inliers,
                confidence=confidence,
                warp_stretch=warp_stretch,
            )

        except Exception as e:
            logger.warning(f"ORB feature extraction failed: {e}")
            return ORBFeatures.empty()

    def extract_hash_features(
        self, query_path: str, reference_path: str
    ) -> HashFeatures:
        """Extract perceptual hash features between two images.

        Args:
            query_path: Path to the query image
            reference_path: Path to the reference image

        Returns:
            HashFeatures with hash distance metrics
        """
        try:
            query_img = Image.open(query_path)
            ref_img = Image.open(reference_path)

            # Perceptual hash (DCT-based)
            q_phash = imagehash.phash(query_img)
            r_phash = imagehash.phash(ref_img)

            # Difference hash (gradient-based)
            q_dhash = imagehash.dhash(query_img)
            r_dhash = imagehash.dhash(ref_img)

            # Wavelet hash (multi-resolution)
            q_whash = imagehash.whash(query_img)
            r_whash = imagehash.whash(ref_img)

            # Average hash
            q_ahash = imagehash.average_hash(query_img)
            r_ahash = imagehash.average_hash(ref_img)

            return HashFeatures(
                phash_distance=float(q_phash - r_phash),
                dhash_distance=float(q_dhash - r_dhash),
                whash_distance=float(q_whash - r_whash),
                ahash_distance=float(q_ahash - r_ahash),
            )

        except Exception as e:
            logger.warning(f"Hash feature extraction failed: {e}")
            return HashFeatures.empty()

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
            # ORB features
            orb_feats = self.extract_orb_features(glashelder_path, wikiart_path)
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

            # Hash features
            hash_feats = self.extract_hash_features(glashelder_path, wikiart_path)
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
