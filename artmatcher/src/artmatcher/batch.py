"""Batch classification for artwork matching across glashelder dataset."""

import contextlib
import csv
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rich.console import Console

from artscraper import Artwork

from .artworks import artwork_of
from .classifier import MLClassifier
from .features import NO_IMAGE_FEATURES
from .features.worker import extract_features_batch, pool  # noqa: F401

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BatchClassifyResult:
    """Result of batch classification for a single image pair."""

    glashelder_path: str
    glashelder_index: int
    artwork_path: str
    artwork_index: int
    similarities: list[float]
    is_match: bool
    confidence: float
    threshold: float
    features: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    """Which of the CLIP top matches this is (0 = the best), see `candidates`."""


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

    def _make_key(self, glashelder_path: str, artwork_path: str) -> str:
        """Create a cache key from image paths."""
        return f"{glashelder_path}||{artwork_path}"

    def get(self, glashelder_path: str, artwork_path: str) -> dict[str, float] | None:
        """Get cached features for an image pair.

        Args:
            glashelder_path: Path to glashelder image
            artwork_path: Path or URL of the reference image

        Returns:
            Feature dict if cached with every current image feature, None otherwise
        """
        features = self._cache.get(self._make_key(glashelder_path, artwork_path))
        if features is None or not NO_IMAGE_FEATURES.keys() <= features.keys():
            return None
        return features

    def set(
        self, glashelder_path: str, artwork_path: str, features: dict[str, float]
    ) -> None:
        """Cache features for an image pair.

        Args:
            glashelder_path: Path to glashelder image
            artwork_path: Path or URL of the reference image
            features: Feature dict to cache
        """
        key = self._make_key(glashelder_path, artwork_path)
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

    def save(self, quiet: bool = False) -> None:
        """Save cache to disk.

        Args:
            quiet: If True, don't print success message (for periodic saves)
        """
        if self.cache_path and self._dirty:
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._cache, f)
                if not quiet:
                    console.print(
                        f"[green]Saved {len(self._cache)} cached features to {self.cache_path}[/green]"
                    )
                self._dirty = False
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def __len__(self) -> int:
        return len(self._cache)

    def __bool__(self) -> bool:
        return True


def predict_confidences(
    classifier: MLClassifier,
    features_by_index: dict[int, dict[str, float]],
    indices: list[int],
) -> list[float]:
    """Vectorized match-confidence prediction for a set of feature dicts."""
    if not indices:
        return []
    return classifier.predict_proba([features_by_index[i] for i in indices]).tolist()


Tried = list[tuple[float, dict[str, float]]]
"""Confidence and features of each candidate tried for an image, in CLIP order."""


def shifted(similarities, rank: int) -> list[float]:
    """The similarities the classifier sees for the candidate at `rank`.

    The list from that candidate on, padded with the last value back to the full
    length: to the classifier, candidate `rank` is the top match, and only the
    ones ranked below it are the competition.
    """
    values = [float(s) for s in similarities]
    return values[rank:] + values[-1:] * rank


def needs_more(tried: Tried, threshold: float, candidates: int, available: int) -> bool:
    """Whether another candidate has to be tried: none matched yet, and N not reached."""
    if len(tried) >= min(candidates, available):
        return False
    return all(confidence <= threshold for confidence, _ in tried)


def choose(tried: Tried, threshold: float, candidates: int) -> int:
    """The rank to go with: the most confident of the candidates tried at this
    threshold, which stops at the first match (so that one, if there is one)."""
    considered = tried[:candidates]
    for rank, (confidence, _) in enumerate(considered):
        if confidence > threshold:
            considered = considered[: rank + 1]
            break
    return max(range(len(considered)), key=lambda rank: considered[rank][0])


def score_round(
    tried: dict[int, Tried],
    images,
    *,
    threshold: float,
    candidates: int,
    glashelder_paths: list[str],
    artwork_paths: list[str],
    top_indices,
    top_similarities,
    classifier: MLClassifier,
    extract,
    lock: contextlib.AbstractContextManager | None = None,
) -> bool:
    """Score the next candidate of every image in `images` that needs one.

    `tried` is updated in place; `extract(work_items)` computes the ORB/hash
    features. Returns whether there was anything left to score, so callers run
    rounds until it says no.

    `lock` guards `tried` for callers that share it between threads. It is not
    held during `extract`; a score another thread added for the same rank in
    the meantime is kept, and this one dropped.
    """
    lock = contextlib.nullcontext() if lock is None else lock
    with lock:
        need = [
            (image, len(tried.get(image, [])))
            for image in images
            if needs_more(
                tried.get(image, []), threshold, candidates, len(top_indices[image])
            )
        ]
    if not need:
        return False

    work_items = [
        (
            n,
            glashelder_paths[image],
            artwork_paths[int(top_indices[image][rank])],
            shifted(top_similarities[image], rank),
            0,
        )
        for n, (image, rank) in enumerate(need)
    ]

    features = extract(work_items)
    confidences = predict_confidences(classifier, features, list(range(len(need))))
    with lock:
        for n, (image, rank) in enumerate(need):
            scores = tried.setdefault(image, [])
            if len(scores) == rank:
                scores.append((confidences[n], features[n]))
    return True


def batch_classify(
    glashelder_paths: list[str],
    artwork_paths: list[str],
    top_indices: np.ndarray,
    top_similarities: np.ndarray,
    classifier: MLClassifier,
    top_k: int = 10,
    threshold: float = 0.5,
    workers: int = 8,
    min_confidence: float | None = None,
    cache: FeatureCache | None = None,
    timeout_per_item: float = 30.0,
    cache_save_interval: int = 100,
    candidates: int = 1,
) -> BatchClassifyOutput:
    """Batch classify all glashelder images against the reference artworks.

    Args:
        glashelder_paths: List of glashelder image paths
        artwork_paths: List of reference artwork paths or URLs
        top_indices: `(glashelder, k)` best references per image, from
            `loading.top_matches`
        top_similarities: their similarities
        classifier: Trained ML classifier
        top_k: Number of top artwork matches to consider
        threshold: Confidence threshold for is_match
        workers: Number of parallel workers for feature extraction
        min_confidence: Only include results above this confidence (None = include all)
        cache: Optional feature cache
        timeout_per_item: Timeout in seconds per item for feature extraction
        cache_save_interval: Save cache to disk every N items (0 to disable)
        candidates: How many of the top matches to classify while the ones
            before them are no match; the most confident one is kept

    Returns:
        BatchClassifyOutput with metadata, summary, and results
    """
    start_time = datetime.now()
    n_glashelder = len(glashelder_paths)
    n_artworks = len(artwork_paths)

    console.print("[bold cyan]Batch Classification[/bold cyan]")
    console.print(f"  Glashelder images: {n_glashelder}")
    console.print(f"  Artwork images: {n_artworks}")
    console.print(f"  Top-k matches: {top_k}")
    console.print(f"  Threshold: {threshold}")
    console.print(f"  Candidates: {candidates}")
    console.print(f"  Workers: {workers}")

    # ORB/hash features and classification, one candidate rank per round: the
    # first round does every image's top match, later ones only the images
    # that have no match yet
    classifier.threshold = threshold
    tried: dict[int, Tried] = {}
    rounds = 0
    while score_round(
        tried,
        range(n_glashelder),
        threshold=threshold,
        candidates=candidates,
        glashelder_paths=glashelder_paths,
        artwork_paths=artwork_paths,
        top_indices=top_indices,
        top_similarities=top_similarities,
        classifier=classifier,
        extract=lambda work_items: extract_features_batch(
            work_items, workers, cache, timeout_per_item, cache_save_interval
        ),
    ):
        rounds += 1
        console.print(f"[green]Scored candidate {rounds} where still needed[/green]")

    # Build results list
    results: list[BatchClassifyResult] = []
    n_matches = 0
    for i in range(n_glashelder):
        rank = choose(tried[i], threshold, candidates)
        confidence, features = tried[i][rank]
        is_match = confidence > threshold
        n_matches += is_match

        # Apply min_confidence filter
        if min_confidence is not None and confidence < min_confidence:
            continue

        artwork_idx = int(top_indices[i, rank])
        result = BatchClassifyResult(
            glashelder_path=glashelder_paths[i],
            glashelder_index=i,
            artwork_path=artwork_paths[artwork_idx],
            artwork_index=artwork_idx,
            similarities=shifted(top_similarities[i], rank),
            is_match=is_match,
            confidence=float(confidence),
            threshold=threshold,
            features=features,
            rank=rank,
        )
        results.append(result)

    console.print(f"[green]Classified {n_glashelder} images[/green]")

    # Build output
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    metadata = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "glashelder_count": n_glashelder,
        "artwork_count": n_artworks,
        "top_k": top_k,
        "threshold": threshold,
        "candidates": candidates,
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


def read_json_output(path: str | Path) -> dict:
    """A JSON file `write_json_output` wrote, also one from before `wikiart_*`
    became `artwork_*`."""
    from .web.store import artwork_keys

    with open(path) as f:
        data = json.load(f)
    data["results"] = [artwork_keys(r) for r in data.get("results", [])]
    return data


def write_json_output(
    output: BatchClassifyOutput,
    path: str | Path,
    artworks: dict[str, Artwork] | None = None,
) -> None:
    """Write batch classification output to JSON file.

    Args:
        output: BatchClassifyOutput to write
        path: Output file path
        artworks: Scraped artwork records, joined onto each result when given
    """
    data = {
        "metadata": output.metadata,
        "summary": output.summary,
        "results": [
            {
                "glashelder_path": r.glashelder_path,
                "glashelder_index": r.glashelder_index,
                "artwork_path": r.artwork_path,
                "artwork_index": r.artwork_index,
                "similarities": r.similarities,
                "rank": r.rank,
                "is_match": r.is_match,
                "confidence": r.confidence,
                "threshold": r.threshold,
                "features": r.features,
                "artwork": asdict(a)
                if (a := artwork_of(r.artwork_path, artworks))
                else None,
            }
            for r in output.results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Wrote JSON output to {path}[/green]")


def write_csv_output(
    output: BatchClassifyOutput,
    path: str | Path,
    artworks: dict[str, Artwork] | None = None,
) -> None:
    """Write batch classification output to CSV file.

    Args:
        output: BatchClassifyOutput to write
        path: Output file path
        artworks: Scraped artwork records, joined onto each result when given
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
        "artwork_path",
        "artwork_index",
        "rank",
        "is_match",
        "confidence",
        "threshold",
        "title",
        "artist",
        "year",
        "artwork_url",
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
            artwork = artwork_of(r.artwork_path, artworks)
            row = [
                r.glashelder_path,
                r.glashelder_index,
                r.artwork_path,
                r.artwork_index,
                r.rank,
                r.is_match,
                r.confidence,
                r.threshold,
                artwork and artwork.title,
                artwork and artwork.artist,
                artwork and artwork.year,
                artwork and artwork.url,
            ]
            row.extend(r.similarities)
            row.extend([r.features.get(fn, 0.0) for fn in feature_names])
            writer.writerow(row)

    console.print(f"[green]Wrote CSV output to {path}[/green]")


def write_output(
    output: BatchClassifyOutput,
    path: str | Path,
    fmt: Literal["json", "csv", "both"],
    artworks: dict[str, Artwork] | None = None,
) -> None:
    """Write batch classification output in the requested format(s).

    `path` is used as given for a single format; for "both" it is written once
    with a `.json` suffix and once with a `.csv` suffix.
    """
    path = Path(path)
    if fmt == "json":
        write_json_output(output, path, artworks=artworks)
    elif fmt == "csv":
        write_csv_output(output, path.with_suffix(".csv"), artworks=artworks)
    else:
        write_json_output(output, path.with_suffix(".json"), artworks=artworks)
        write_csv_output(output, path.with_suffix(".csv"), artworks=artworks)
