"""Batch classification for artwork matching across glashelder dataset."""

import csv
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from image_matching.ml_classifier import (
    MLClassifier,
    _extract_orb_worker,
    _init_orb_worker,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BatchClassifyResult:
    """Result of batch classification for a single image pair."""

    glashelder_path: str
    glashelder_index: int
    wikiart_path: str
    wikiart_index: int
    similarities: list[float]
    is_match: bool
    confidence: float
    threshold: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class BatchClassifyOutput:
    """Complete output from batch classification."""

    metadata: dict[str, Any]
    summary: dict[str, Any]
    results: list[BatchClassifyResult]


class FeatureCache:
    """Optional disk-based cache for ORB/hash features."""

    def __init__(self, cache_path: str | Path | None = None):
        """Initialize the feature cache.

        Args:
            cache_path: Path to save/load cache. If None, caching is disabled.
        """
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict[str, float]] = {}
        self._dirty = False

        if self.cache_path and self.cache_path.exists():
            self._load()

    def _make_key(self, glashelder_path: str, wikiart_path: str) -> str:
        """Create a cache key from image paths."""
        return f"{glashelder_path}||{wikiart_path}"

    def get(self, glashelder_path: str, wikiart_path: str) -> dict[str, float] | None:
        """Get cached features for an image pair.

        Args:
            glashelder_path: Path to glashelder image
            wikiart_path: Path to wikiart image

        Returns:
            Feature dict if cached, None otherwise
        """
        key = self._make_key(glashelder_path, wikiart_path)
        return self._cache.get(key)

    def set(
        self, glashelder_path: str, wikiart_path: str, features: dict[str, float]
    ) -> None:
        """Cache features for an image pair.

        Args:
            glashelder_path: Path to glashelder image
            wikiart_path: Path to wikiart image
            features: Feature dict to cache
        """
        key = self._make_key(glashelder_path, wikiart_path)
        self._cache[key] = features
        self._dirty = True

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                console.print(
                    f"[cyan]Loaded {len(self._cache)} cached features from {self.cache_path}[/cyan]"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def save(self) -> None:
        """Save cache to disk."""
        if self.cache_path and self._dirty:
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._cache, f)
                console.print(
                    f"[green]Saved {len(self._cache)} cached features to {self.cache_path}[/green]"
                )
                self._dirty = False
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def __len__(self) -> int:
        return len(self._cache)


def compute_embedding_similarities(
    glashelder_emb: torch.Tensor,
    wikiart_emb: torch.Tensor,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k similar wikiart images for each glashelder image.

    Args:
        glashelder_emb: Glashelder embeddings tensor (N x D)
        wikiart_emb: Wikiart embeddings tensor (M x D)
        top_k: Number of top matches to return per image

    Returns:
        Tuple of (indices, similarities) arrays, each shape (N, top_k)
    """
    # Normalize embeddings
    glashelder_norm = glashelder_emb / glashelder_emb.norm(dim=-1, keepdim=True)
    wikiart_norm = wikiart_emb / wikiart_emb.norm(dim=-1, keepdim=True)

    # Compute all pairwise similarities (N x M)
    similarities = glashelder_norm @ wikiart_norm.T

    # Get top-k for each glashelder image
    actual_k = min(top_k, wikiart_emb.shape[0])
    top_values, top_indices = torch.topk(similarities, actual_k, dim=1)

    return top_indices.cpu().numpy(), top_values.cpu().numpy()


def _get_default_features(similarities: list[float]) -> dict[str, float]:
    """Return default features when extraction fails."""
    from image_matching.ml_classifier import _extract_distribution_features

    features = _extract_distribution_features(similarities)
    # Add default ORB features
    features["orb_num_matches"] = 0.0
    features["orb_num_inliers"] = 0.0
    features["orb_confidence"] = 0.0
    features["orb_warp_stretch"] = 1.0
    features["inlier_match_ratio"] = 0.0
    # Add default hash features (max distance = no similarity)
    features["phash_distance"] = 64.0
    features["dhash_distance"] = 64.0
    features["whash_distance"] = 64.0
    features["ahash_distance"] = 64.0
    return features


def extract_features_batch(
    work_items: list[tuple[int, str, str, list[float], int]],
    workers: int,
    cache: FeatureCache | None = None,
    timeout_per_item: float = 30.0,
) -> dict[int, dict[str, float]]:
    """Extract ORB/hash features for a batch of image pairs in parallel.

    Args:
        work_items: List of (index, glashelder_path, wikiart_path, similarities, label)
        workers: Number of parallel workers
        cache: Optional feature cache
        timeout_per_item: Timeout in seconds per item (default 30s)

    Returns:
        Dict mapping index to feature dict
    """
    results: dict[int, dict[str, float]] = {}

    # Check cache first
    uncached_items = []
    for item in work_items:
        idx, glashelder_path, wikiart_path, _, _ = item
        if cache:
            cached_features = cache.get(glashelder_path, wikiart_path)
            if cached_features is not None:
                results[idx] = cached_features
                continue
        uncached_items.append(item)

    if not uncached_items:
        return results

    # Extract features in parallel for uncached items
    n_failed = 0
    n_completed = 0

    # Calculate a reasonable global timeout
    # Allow timeout_per_item seconds per item, but at least 5 minutes
    global_timeout = max(300, len(uncached_items) * timeout_per_item / workers)

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
            total=len(uncached_items),
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_orb_worker,
        ) as executor:
            future_to_item = {
                executor.submit(_extract_orb_worker, item): item
                for item in uncached_items
            }

            try:
                for future in as_completed(future_to_item, timeout=global_timeout):
                    item = future_to_item[future]
                    idx = item[0]
                    similarities = item[3]

                    try:
                        # Add timeout to prevent hanging on result retrieval
                        _, features, _ = future.result(timeout=timeout_per_item)
                        results[idx] = features

                        # Update cache
                        if cache:
                            cache.set(item[1], item[2], features)
                    except TimeoutError:
                        logger.warning(f"Timeout extracting features for {item[1]}")
                        results[idx] = _get_default_features(similarities)
                        n_failed += 1
                    except Exception as e:
                        logger.warning(f"Error extracting features for {item[1]}: {e}")
                        results[idx] = _get_default_features(similarities)
                        n_failed += 1

                    n_completed += 1
                    progress.advance(task)

            except TimeoutError:
                # Global timeout - cancel remaining futures and use defaults
                console.print(
                    f"[yellow]Global timeout reached after {n_completed} items[/yellow]"
                )
                for future, item in future_to_item.items():
                    if not future.done():
                        future.cancel()
                        idx = item[0]
                        similarities = item[3]
                        if idx not in results:
                            results[idx] = _get_default_features(similarities)
                            n_failed += 1
                            progress.advance(task)

    if n_failed > 0:
        console.print(
            f"[yellow]Warning: {n_failed} items failed feature extraction[/yellow]"
        )

    return results


def batch_classify(
    glashelder_emb: torch.Tensor,
    glashelder_paths: list[str],
    wikiart_emb: torch.Tensor,
    wikiart_paths: list[str],
    classifier: MLClassifier,
    top_k: int = 10,
    threshold: float = 0.5,
    workers: int = 8,
    min_confidence: float | None = None,
    cache: FeatureCache | None = None,
    timeout_per_item: float = 30.0,
) -> BatchClassifyOutput:
    """Batch classify all glashelder images against wikiart.

    Args:
        glashelder_emb: Glashelder embeddings tensor
        glashelder_paths: List of glashelder image paths
        wikiart_emb: Wikiart embeddings tensor
        wikiart_paths: List of wikiart image paths
        classifier: Trained ML classifier
        top_k: Number of top wikiart matches to consider
        threshold: Confidence threshold for is_match
        workers: Number of parallel workers for feature extraction
        min_confidence: Only include results above this confidence (None = include all)
        cache: Optional feature cache
        timeout_per_item: Timeout in seconds per item for feature extraction

    Returns:
        BatchClassifyOutput with metadata, summary, and results
    """
    start_time = datetime.now()
    n_glashelder = len(glashelder_paths)
    n_wikiart = len(wikiart_paths)

    console.print("[bold cyan]Batch Classification[/bold cyan]")
    console.print(f"  Glashelder images: {n_glashelder}")
    console.print(f"  Wikiart images: {n_wikiart}")
    console.print(f"  Top-k matches: {top_k}")
    console.print(f"  Threshold: {threshold}")
    console.print(f"  Workers: {workers}")

    # Phase 1: Compute embedding similarities
    console.print("\n[cyan]Phase 1: Computing embedding similarities...[/cyan]")
    top_indices, top_sims = compute_embedding_similarities(
        glashelder_emb, wikiart_emb, top_k
    )
    console.print(f"[green]Computed similarities for {n_glashelder} images[/green]")

    # Phase 2: Prepare work items for ORB/hash extraction
    console.print("\n[cyan]Phase 2: Extracting ORB/hash features...[/cyan]")
    work_items = []
    for i in range(n_glashelder):
        glashelder_path = glashelder_paths[i]
        wikiart_idx = int(top_indices[i, 0])
        wikiart_path = wikiart_paths[wikiart_idx]
        similarities = top_sims[i].tolist()
        work_items.append((i, glashelder_path, wikiart_path, similarities, 0))

    # Extract features in parallel
    features_map = extract_features_batch(work_items, workers, cache, timeout_per_item)

    # Phase 3: ML Classification
    console.print("\n[cyan]Phase 3: Running ML classification...[/cyan]")
    classifier.threshold = threshold

    results: list[BatchClassifyResult] = []
    n_matches = 0

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
            "[cyan]Classifying...",
            total=n_glashelder,
        )

        for i in range(n_glashelder):
            features = features_map[i]
            similarities = top_sims[i].tolist()

            # Build feature vector
            feature_vector = np.array(
                [features.get(f, 0.0) for f in classifier.FEATURE_ORDER]
            ).reshape(1, -1)

            # Get prediction
            proba = classifier._model.predict_proba(feature_vector)[0]  # type:ignore
            confidence = float(proba[1])
            is_match = confidence >= threshold

            if is_match:
                n_matches += 1

            # Apply min_confidence filter
            if min_confidence is not None and confidence < min_confidence:
                progress.advance(task)
                continue

            wikiart_idx = int(top_indices[i, 0])
            result = BatchClassifyResult(
                glashelder_path=glashelder_paths[i],
                glashelder_index=i,
                wikiart_path=wikiart_paths[wikiart_idx],
                wikiart_index=wikiart_idx,
                similarities=similarities,
                is_match=is_match,
                confidence=confidence,
                threshold=threshold,
                features=features,
            )
            results.append(result)
            progress.advance(task)

    # Build output
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    metadata = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "glashelder_count": n_glashelder,
        "wikiart_count": n_wikiart,
        "top_k": top_k,
        "threshold": threshold,
        "workers": workers,
        "min_confidence": min_confidence,
    }

    match_rate = n_matches / n_glashelder if n_glashelder > 0 else 0.0
    summary = {
        "total_processed": n_glashelder,
        "total_matches": n_matches,
        "match_rate": match_rate,
        "results_returned": len(results),
    }

    console.print("\n[bold green]Classification complete![/bold green]")
    console.print(f"  Total matches: {n_matches} ({match_rate:.2%})")
    console.print(f"  Results returned: {len(results)}")
    console.print(f"  Duration: {duration:.1f}s")

    return BatchClassifyOutput(
        metadata=metadata,
        summary=summary,
        results=results,
    )


def write_json_output(output: BatchClassifyOutput, path: str | Path) -> None:
    """Write batch classification output to JSON file.

    Args:
        output: BatchClassifyOutput to write
        path: Output file path
    """
    data = {
        "metadata": output.metadata,
        "summary": output.summary,
        "results": [
            {
                "glashelder_path": r.glashelder_path,
                "glashelder_index": r.glashelder_index,
                "wikiart_path": r.wikiart_path,
                "wikiart_index": r.wikiart_index,
                "similarities": r.similarities,
                "is_match": r.is_match,
                "confidence": r.confidence,
                "threshold": r.threshold,
                "features": r.features,
            }
            for r in output.results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Wrote JSON output to {path}[/green]")


def write_csv_output(output: BatchClassifyOutput, path: str | Path) -> None:
    """Write batch classification output to CSV file.

    Args:
        output: BatchClassifyOutput to write
        path: Output file path
    """
    if not output.results:
        console.print("[yellow]No results to write to CSV[/yellow]")
        return

    # Get all feature names from first result
    feature_names = list(output.results[0].features.keys())

    # Build header
    header = [
        "glashelder_path",
        "glashelder_index",
        "wikiart_path",
        "wikiart_index",
        "is_match",
        "confidence",
        "threshold",
    ]
    # Add similarity columns
    n_sims = len(output.results[0].similarities)
    header.extend([f"sim_{i}" for i in range(n_sims)])
    # Add feature columns
    header.extend(feature_names)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for r in output.results:
            row = [
                r.glashelder_path,
                r.glashelder_index,
                r.wikiart_path,
                r.wikiart_index,
                r.is_match,
                r.confidence,
                r.threshold,
            ]
            row.extend(r.similarities)
            row.extend([r.features.get(fn, 0.0) for fn in feature_names])
            writer.writerow(row)

    console.print(f"[green]Wrote CSV output to {path}[/green]")
