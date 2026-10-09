"""Process-pool extraction of ORB/hash features, with a per-item timeout."""

import contextlib
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.progress import TaskID

from common.progress import progress as progress_bars

from .image import NO_IMAGE_FEATURES, _pair_features, make_orb
from .similarity import _sanitize_features, distribution_features

logger = logging.getLogger(__name__)
console = Console()

# Global variables for worker processes
_worker_orb = None
_worker_bf = None


def _init_orb_worker() -> None:
    """Initialize worker process with ORB detector and BFMatcher."""
    global _worker_orb, _worker_bf
    _worker_orb, _worker_bf = make_orb()


def _extract_orb_worker(
    args: tuple[int, str, str, list[float], int], timeout: float = 25
) -> tuple[int, dict[str, float], int]:
    """Worker function to extract features for a single entry.

    Args:
        args: Tuple of (index, glashelder_path, artwork_path, similarities, label)
        timeout: Seconds before giving up on the images and returning only the
            distribution features

    Returns:
        Tuple of (index, features_dict, label)
    """
    global _worker_orb, _worker_bf

    def handler(_signum, _frame):
        raise TimeoutError("Worker timeout")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        idx, glashelder_path, artwork_path, similarities, label = args

        # Initialize if needed (shouldn't happen with initializer, but be safe)
        if _worker_orb is None or _worker_bf is None:
            _init_orb_worker()

        features = (
            distribution_features(similarities)
            | _pair_features(
                glashelder_path, artwork_path, _worker_orb, _worker_bf
            ).as_dict()
        )
        return (idx, _sanitize_features(features), label)

    except TimeoutError:
        idx, glashelder_path, _, similarities, label = args
        logger.warning(f"Worker timeout for {glashelder_path}")
        features = distribution_features(similarities) | NO_IMAGE_FEATURES
        return (idx, _sanitize_features(features), label)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class _NullProgress:
    """No-op stand-in for rich.progress.Progress when progress output is disabled."""

    def add_task(self, *_args: Any, **_kwargs: Any) -> TaskID:
        return TaskID(0)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass


def pool(workers: int) -> ProcessPoolExecutor:
    """Worker processes for ORB/hash extraction, each with its own detector."""
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_orb_worker)


def _get_default_features(similarities: list[float]) -> dict[str, float]:
    """Return default features when extraction fails."""
    return distribution_features(similarities) | NO_IMAGE_FEATURES


def extract_features_batch(
    work_items: list[tuple[int, str, str, list[float], int]],
    workers: int,
    cache: Any | None = None,
    timeout_per_item: float = 30.0,
    cache_save_interval: int = 100,
    progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> dict[int, dict[str, float]]:
    """Extract ORB/hash features for a batch of image pairs in parallel.

    Args:
        work_items: List of (index, glashelder_path, artwork_path, similarities, label)
        workers: Number of parallel workers
        cache: Optional feature cache (a `batch.FeatureCache` or alike)
        timeout_per_item: Seconds a worker spends on one item before it falls
            back to the distribution features only (default 30s)
        cache_save_interval: Save cache to disk every N items (default 100)
        progress: If False, suppress the progress bar and all console output
        executor: A pool to use and leave running, for callers that extract
            often (one per call costs a fork of every worker each time). Made
            from `pool(workers)` and shut down here when not given.

    Returns:
        Dict mapping index to feature dict
    """
    results: dict[int, dict[str, float]] = {}

    # Check cache first
    uncached_items = []
    for item in work_items:
        idx, glashelder_path, artwork_path, _, _ = item
        if cache:
            cached_features = cache.get(glashelder_path, artwork_path)
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

    progress_cm = (
        progress_bars(console=console)
        if progress
        else contextlib.nullcontext(_NullProgress())
    )

    with progress_cm as progress_bar:
        task = progress_bar.add_task(
            "[cyan]Extracting ORB/hash features...",
            total=len(uncached_items),
        )

        owned = executor is None
        if executor is None:
            executor = pool(workers)

        try:
            # the worker itself gives up on an item after `timeout_per_item`
            future_to_item = {
                executor.submit(_extract_orb_worker, item, timeout_per_item): item
                for item in uncached_items
            }
            for future in as_completed(future_to_item, timeout=global_timeout):
                item = future_to_item[future]
                idx = item[0]
                similarities = item[3]

                try:
                    _, features, _ = future.result()
                    results[idx] = features

                    # Update cache
                    if cache:
                        cache.set(item[1], item[2], features)
                except Exception as e:
                    logger.warning(f"Error extracting features for {item[1]}: {e}")
                    results[idx] = _get_default_features(similarities)
                    n_failed += 1

                n_completed += 1
                progress_bar.advance(task)

                # Periodically save cache to disk for crash recovery
                if (
                    cache
                    and cache_save_interval > 0
                    and n_completed % cache_save_interval == 0
                ):
                    cache.save(quiet=True)

        except TimeoutError:
            # Global timeout - cancel remaining futures and use defaults
            if progress:
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
                        progress_bar.advance(task)

            # Save cache on timeout for crash recovery
            if cache:
                cache.save()
        finally:
            if owned:
                executor.shutdown(wait=False, cancel_futures=True)

    if n_failed > 0 and progress:
        console.print(
            f"[yellow]Warning: {n_failed} items failed feature extraction[/yellow]"
        )

    return results
