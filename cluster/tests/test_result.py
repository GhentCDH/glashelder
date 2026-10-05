"""Checks for saving and loading clustering results.

Run with `uv run python cluster/tests/test_result.py`.
"""

import tempfile
from pathlib import Path

import numpy as np

from cluster.result import (
    Annotations,
    Clustering,
    fingerprint,
    load,
    read_annotations,
    read_labels,
    save,
    status,
    write_annotations,
)
from embed.store import Store

MODEL_ID = "test/model"


def store_with(path: Path, refs: list[str]) -> Store:
    store = Store.create_or_open(path, model="test", model_id=MODEL_ID)
    store.put(refs, [None] * len(refs), np.random.rand(len(refs), 8))
    return store


def clustering_of(store: Store, created: str = "2026-01-01T00:00:00") -> Clustering:
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
            "created": created,
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


def test_annotations():
    with tempfile.TemporaryDirectory() as directory:
        store = store_with(Path(directory) / "e.store", ["a.jpg", "b.jpg"])
        path = Path(directory) / "c.clusters"
        save(path, clustering_of(store))
        assert read_annotations(path) == Annotations()

        written = Annotations(
            names={2: "portraits", 0: "landscapes"}, groups={"people": [2, 0]}
        )
        write_annotations(path, written)
        assert read_annotations(path) == Annotations(
            names={0: "landscapes", 2: "portraits"}, groups={"people": [0, 2]}
        )

        # annotations of another clustering in the same directory do not apply
        file = path / "annotations.json"
        file.write_text(file.read_text().replace("2026-01-01", "1999-01-01"))
        assert read_annotations(path) == Annotations()


def test_save_over_a_result_moves_annotations_and_report():
    with tempfile.TemporaryDirectory() as directory:
        store = store_with(Path(directory) / "e.store", ["a.jpg", "b.jpg"])
        path = Path(directory) / "c.clusters"
        save(path, clustering_of(store))
        write_annotations(path, Annotations(names={1: "portraits"}))
        (path / "report.pdf").write_bytes(b"%PDF")

        save(path, clustering_of(store, created="2026-02-02T00:00:00"))
        assert not (path / "report.pdf").exists()
        assert not (path / "annotations.json").exists()
        assert (path / "annotations-2026-01-01T000000.json").is_file()
        assert read_annotations(path) == Annotations()


if __name__ == "__main__":
    test_round_trip()
    test_refuses_other_directories()
    test_status()
    test_annotations()
    test_save_over_a_result_moves_annotations_and_report()
    print("ok")
