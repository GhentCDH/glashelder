"""Artwork recognition using SIFT features and FLANN matching."""

import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

from image_matching.stitching import (
    compute_homography,
    create_flann_matcher,
    extract_features,
    find_matches_flann,
    load_image,
    resize_max,
)


@dataclass
class MatchResult:
    """Result of matching a query image against a database painting."""

    painting_path: str
    num_matches: int
    num_inliers: int
    confidence: float
    homography: np.ndarray | None

    @property
    def painting_name(self) -> str:
        """Get just the filename of the matched painting."""
        return Path(self.painting_path).name


@dataclass
class RecognitionDatabase:
    """Database of SIFT features for known paintings."""

    features: dict[str, tuple]  # path -> (keypoints, descriptors)
    metadata: dict[str, dict]  # path -> {width, height, etc.}

    def save(self, path: str | Path) -> None:
        """Save database to disk."""
        # Convert keypoints to serializable format
        serializable_features = {}
        for img_path, (kp, des) in self.features.items():
            # Store keypoint data as tuples
            kp_data = [(k.pt, k.size, k.angle, k.response, k.octave) for k in kp]
            serializable_features[img_path] = (kp_data, des)

        data = {
            "features": serializable_features,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "RecognitionDatabase":
        """Load database from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Reconstruct keypoints from serializable format
        features = {}
        for img_path, (kp_data, des) in data["features"].items():
            kp = [
                cv2.KeyPoint(
                    x=pt[0],
                    y=pt[1],
                    size=size,
                    angle=angle,
                    response=response,
                    octave=octave,
                )
                for pt, size, angle, response, octave in kp_data
            ]
            features[img_path] = (kp, des)

        return cls(features=features, metadata=data["metadata"])


# Global variables for worker processes (initialized once per worker)
_worker_sift = None
_worker_max_size = None


def _init_worker(max_size: int | None = None, max_features: int | None = None) -> None:
    """Initialize worker process with a SIFT detector and optional max size.

    Args:
        max_size: Optional maximum dimension for resizing images before feature extraction.
        max_features: Maximum number of keypoints to extract per image (None = unlimited).
    """
    global _worker_sift, _worker_max_size
    # nfeatures=0 means unlimited (default), otherwise limits to best N features
    _worker_sift = cv2.SIFT_create(nfeatures=max_features or 0)  # type:ignore
    _worker_max_size = max_size


def _extract_features_worker(path: str) -> tuple[str, tuple | None, dict | None]:
    """Worker function to extract features from a single image.

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (path, (keypoints, descriptors), metadata) or (path, None, None) on failure.
    """
    try:
        # Use the pre-initialized SIFT detector and max_size from worker process
        global _worker_sift, _worker_max_size
        if _worker_sift is None:
            _worker_sift = cv2.SIFT_create()  # type:ignore

        img = load_image(path)
        h_orig, w_orig = img.shape[:2]

        # Resize if max_size is set
        if _worker_max_size is not None:
            img = resize_max(img, _worker_max_size)

        kp, des = extract_features(img, _worker_sift)

        if des is None or len(kp) == 0:
            return (path, None, None)

        # Serialize keypoints for pickling across process boundary
        kp_data = [(k.pt, k.size, k.angle, k.response, k.octave) for k in kp]

        # Store original dimensions in metadata
        metadata = {
            "width": w_orig,
            "height": h_orig,
            "num_features": len(kp),
            "resized": _worker_max_size is not None,
        }
        return (path, (kp_data, des), metadata)
    except Exception:
        return (path, None, None)


def build_database(
    painting_paths: list[str],
    *,
    workers: int = 1,
    max_size: int | None = None,
    max_features: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> RecognitionDatabase:
    """Build a database of SIFT features for known paintings.

    Args:
        painting_paths: List of paths to painting images.
        workers: Number of parallel workers for feature extraction (default: 1 for sequential).
        max_size: Optional maximum dimension for resizing images before feature extraction.
        max_features: Maximum number of keypoints to extract per image. Limits database size.
                     Recommended: 500-2000 for recognition tasks. None = unlimited (large databases).
        on_progress: Optional callback for progress messages.

    Returns:
        RecognitionDatabase containing features and metadata.
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    features = {}
    metadata = {}

    log(
        f"Extracting features from {len(painting_paths)} paintings using {workers} worker(s)..."
    )

    if workers <= 1:
        # Sequential processing (original behavior)
        sift = cv2.SIFT_create(nfeatures=max_features or 0)  # type:ignore
        for path in track(painting_paths):
            img = load_image(path)
            h_orig, w_orig = img.shape[:2]

            # Resize if max_size is set
            if max_size is not None:
                img = resize_max(img, max_size)

            kp, des = extract_features(img, sift)

            if des is None or len(kp) == 0:
                log(f"  Warning: No features found in {Path(path).name}")
                continue

            features[path] = (kp, des)
            metadata[path] = {
                "width": w_orig,
                "height": h_orig,
                "num_features": len(kp),
                "resized": max_size is not None,
            }
    else:
        # Parallel processing using multiprocessing
        console = Console()
        warnings = []

        console.print(f"[cyan]Starting {workers} worker processes...[/cyan]")

        try:
            with Progress(console=console, refresh_per_second=10) as progress:
                task = progress.add_task(
                    "[cyan]Extracting features...", total=len(painting_paths)
                )

                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(max_size, max_features),
                ) as executor:
                    try:
                        # Submit all tasks and process as they complete (not in order)
                        future_to_idx = {
                            executor.submit(_extract_features_worker, path): idx
                            for idx, path in enumerate(painting_paths)
                        }

                        # Process results as workers complete them (async)
                        for future in as_completed(future_to_idx):
                            path, result, meta = future.result()

                            if result is None or meta is None:
                                warnings.append(Path(path).name)
                            else:
                                kp_data, des = result
                                kp = [
                                    cv2.KeyPoint(
                                        x=pt[0],
                                        y=pt[1],
                                        size=size,
                                        angle=angle,
                                        response=response,
                                        octave=octave,
                                    )
                                    for pt, size, angle, response, octave in kp_data
                                ]
                                features[path] = (kp, des)
                                metadata[path] = meta

                            progress.advance(task, 1)
                    except KeyboardInterrupt:
                        console.print(
                            "\n[yellow]Interrupted by user, cleaning up...[/yellow]"
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

        except KeyboardInterrupt:
            console.print("[red]Build cancelled by user[/red]")
            raise

        # Show warnings after progress completes
        if warnings:
            console.print(
                f"\n[yellow]Warning: No features found in {len(warnings)} image(s):[/yellow]"
            )
            for name in warnings:
                console.print(f"  - {name}")

    log(f"Database built with {len(features)} paintings")
    return RecognitionDatabase(features=features, metadata=metadata)


def find_painting(
    query_path: str,
    database: RecognitionDatabase,
    *,
    min_matches: int = 10,
    min_inliers: int = 4,
    max_results: int = 5,
    verify_geometry: bool = True,
) -> list[MatchResult]:
    """Find which painting(s) the query image matches.

    Args:
        query_path: Path to query image (e.g., photo of a painting).
        database: Pre-built recognition database.
        min_matches: Minimum number of good matches to consider.
        min_inliers: Minimum number of inliers after homography verification.
        max_results: Maximum number of results to return.
        verify_geometry: Whether to verify with homography (recommended).

    Returns:
        List of MatchResult objects, sorted by confidence (best first).
    """
    # Extract features from query
    query_img = load_image(query_path)
    sift = cv2.SIFT_create()  # type:ignore
    query_kp, query_des = extract_features(query_img, sift)

    if query_des is None or len(query_kp) == 0:
        return []

    # Match against all paintings in database
    flann = create_flann_matcher()
    results = []

    for painting_path, (painting_kp, painting_des) in database.features.items():
        # Find matches
        matches = find_matches_flann(query_des, painting_des, flann)

        if len(matches) < min_matches:
            continue

        # Verify geometry with homography
        num_inliers = 0
        h = None

        if verify_geometry and len(matches) >= 4:
            h = compute_homography(query_kp, painting_kp, matches)
            if h is not None:
                # Count inliers (matches that fit the homography well)
                src_pts = np.float32(
                    [query_kp[m.queryIdx].pt for m in matches]  # type:ignore
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [painting_kp[m.trainIdx].pt for m in matches]  # type:ignore
                ).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                num_inliers = int(mask.ravel().sum()) if mask is not None else 0

                if num_inliers < min_inliers:
                    continue
            else:
                # Homography computation failed
                continue
        else:
            # No geometry verification - use match count
            num_inliers = len(matches)

        # Compute confidence score
        # Higher is better: ratio of inliers to query features
        confidence = num_inliers / len(query_kp) if len(query_kp) > 0 else 0.0

        results.append(
            MatchResult(
                painting_path=painting_path,
                num_matches=len(matches),
                num_inliers=num_inliers,
                confidence=confidence,
                homography=h,
            )
        )

    # Sort by confidence (best first)
    results.sort(key=lambda r: r.confidence, reverse=True)

    return results[:max_results]
