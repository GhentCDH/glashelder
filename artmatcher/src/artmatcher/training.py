"""Reading a labeled dataset and extracting its features for training."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from common.progress import progress as progress_bars

from .features.image import NO_IMAGE_FEATURES, _pair_features, make_orb
from .features.similarity import _sanitize_features, distribution_features
from .features.worker import extract_features_batch

console = Console()


def read_labeled_entries(dataset_path: str | Path) -> list[tuple[dict, int]]:
    """The labeled entries of a `labeling_dataset.json`, as `(entry, label)` pairs."""
    from .web.store import normalize_dataset

    console.print(f"[cyan]Loading dataset from {dataset_path}...[/cyan]")
    with open(dataset_path) as f:
        dataset = normalize_dataset(json.load(f))

    return [(entry, int(truth)) for entry, truth in dataset.labeled()]


def _entry_features(entry: dict, extract_image_features: bool, orb_bf: tuple) -> dict:
    """Features of a single dataset entry, computed in this process."""
    similarities = entry.get("similarities", [])
    glashelder_path = entry.get("glashelder_path") if extract_image_features else None
    artwork_path = entry.get("artwork_path") if extract_image_features else None

    image_features = (
        _pair_features(glashelder_path, artwork_path, *orb_bf).as_dict()
        if glashelder_path and artwork_path
        else NO_IMAGE_FEATURES
    )
    return _sanitize_features(distribution_features(similarities) | image_features)


def _extract_features_sequential(
    entries_with_labels: list[tuple[dict, int]],
    extract_image_features: bool,
) -> tuple[list[dict], list[int]]:
    """Extract features sequentially with a progress bar."""
    orb_bf = make_orb()
    features_list = []
    labels_list = []

    with progress_bars(console=console) as progress:
        task = progress.add_task(
            "[cyan]Extracting features...",
            total=len(entries_with_labels),
        )

        for entry, label in entries_with_labels:
            features_list.append(_entry_features(entry, extract_image_features, orb_bf))
            labels_list.append(label)
            progress.advance(task)

    return features_list, labels_list


def _extract_features_parallel(
    entries_with_labels: list[tuple[dict, int]],
    workers: int,
) -> tuple[list[dict], list[int]]:
    """Extract features in parallel, reusing the shared worker pool extraction."""
    work_items = [
        (
            idx,
            entry.get("glashelder_path", ""),
            entry.get("artwork_path") or "",
            entry.get("similarities", []),
            label,
        )
        for idx, (entry, label) in enumerate(entries_with_labels)
    ]

    console.print(
        f"[cyan]Starting {workers} worker processes for ORB/hash extraction...[/cyan]"
    )
    features_by_index = extract_features_batch(work_items, workers)

    features_list = [features_by_index[idx] for idx in range(len(work_items))]
    labels_list = [label for _, label in entries_with_labels]
    return features_list, labels_list


def extract_training_features(
    entries_with_labels: list[tuple[dict, int]],
    extract_image_features: bool,
    workers: int,
    features_to_array,
) -> tuple[NDArray, NDArray]:
    """Feature matrix and label array for a labeled dataset.

    `features_to_array` orders each feature dict the same way the classifier does.
    """
    if extract_image_features and workers > 1:
        features_list, labels_list = _extract_features_parallel(
            entries_with_labels, workers
        )
    else:
        features_list, labels_list = _extract_features_sequential(
            entries_with_labels, extract_image_features
        )

    features_array = np.array([features_to_array(f) for f in features_list])
    labels_array = np.array(labels_list)
    return features_array, labels_array
