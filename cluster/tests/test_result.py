"""Checks for saving and loading clustering results.

Run with `uv run python cluster/tests/test_result.py`.
"""

import tempfile
from pathlib import Path

import numpy as np

from cluster.result import Clustering, fingerprint, load, read_labels, save, status
from embed.store import Store

MODEL_ID = "test/model"


def store_with(path: Path, refs: list[str]) -> Store:
    store = Store.create_or_open(path, model="test", model_id=MODEL_ID)
    store.put(refs, [None] * len(refs), np.random.rand(len(refs), 8))
    return store


def clustering_of(store: Store) -> Clustering:
    n = len(store)
    rng = np.random.default_rng(0)
    return Clustering(
        refs=store.refs(),
        labels=rng.integers(-1, 3, n),
        probabilities=rng.random(n),
        outlier_scores=rng.random(n),
        coords2d=rng.random((n, 2)),
        coords3d=rng.random((n, 3)),
        reduced=rng.random((n, 30)),
        meta={
            "model_id": store.model_id,
            "rows": n,
            "edits": store.edits,
            "refs_hash": fingerprint(store.refs(), store.edits),
        },
    )


def test_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        store = store_with(Path(directory) / "e.store", ["a.jpg", "b/c.jpg", "d.jpg"])
        saved = clustering_of(store)
        save(Path(directory) / "c.clusters", saved)
        loaded = load(Path(directory) / "c.clusters")

        assert loaded.refs == saved.refs
        assert loaded.meta["version"] == 1
        assert np.array_equal(loaded.labels, saved.labels)
        assert np.array_equal(read_labels(Path(directory) / "c.clusters"), saved.labels)
        for name in ("probabilities", "outlier_scores", "coords2d", "coords3d"):
            assert np.allclose(getattr(loaded, name), getattr(saved, name))
        assert loaded.reduced.shape == (3, 30)
        assert np.allclose(loaded.reduced, saved.reduced)


def test_refuses_other_directories():
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "something").touch()
        store = store_with(Path(directory) / "e.store", ["a.jpg"])
        try:
            save(directory, clustering_of(store))
        except ValueError:
            return
        raise AssertionError("saved over a directory that is no clustering")


def test_status():
    with tempfile.TemporaryDirectory() as directory:
        store = store_with(Path(directory) / "e.store", ["a.jpg", "b.jpg"])
        meta = clustering_of(store).meta
        assert status(meta, store) == "up to date"

        store.put(["c.jpg"], [None], np.random.rand(1, 8))
        assert status(meta, store).startswith("incomplete: 1 rows")

        store.put(["a.jpg"], [None], np.random.rand(1, 8))
        assert status(meta, store).startswith("stale")


if __name__ == "__main__":
    test_round_trip()
    test_refuses_other_directories()
    test_status()
    print("ok")
