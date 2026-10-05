"""A clustering of an embedding store, saved so another machine can use it.

A result is a directory:

    meta.json       what was clustered (the embedding store, its row count, its
                    `edits` and a fingerprint of its refs) and with which parameters
    points.parquet  one row per embedding row: ref, label, scores and coordinates

Parquet, so anything reads it (polars, pandas, DuckDB, a browser) without this
package. Refs are kept as the embedding store has them: relative to its image
root, or URLs.

A result covers the first `rows` rows of its store. Rows appended later do not
change those, so the result stays right for them and only misses the new ones.
Rows changed in place or removed (the store's `edits`) make it stale.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embed.neighbours import queries_hash
from embed.store import Store

META, POINTS = "meta.json", "points.parquet"
VERSION = 1


@dataclass
class Clustering:
    refs: list[str]
    """Line i belongs to row i of the embedding store."""
    labels: np.ndarray
    """`(n,)` HDBSCAN labels, -1 is noise."""
    probabilities: np.ndarray
    """`(n,)` how strongly each point belongs to its cluster."""
    outlier_scores: np.ndarray
    """`(n,)` GLOSH outlier score, higher is more of an outlier."""
    coords2d: np.ndarray
    coords3d: np.ndarray
    reduced: np.ndarray
    """`(n, d)` the UMAP reduction HDBSCAN ran on."""
    meta: dict


def fingerprint(refs: list[str], edits: int | None) -> str:
    return queries_hash(refs, edits or 0)


def is_clustering(path: str | Path) -> bool:
    return (Path(path) / META).is_file()


def save(path: str | Path, clustering: Clustering) -> None:
    """Write `clustering` to `path`, replacing a result that is already there."""
    path = Path(path)
    if path.exists() and not is_clustering(path):
        if path.is_file() or any(path.iterdir()):
            raise ValueError(f"{path} exists and is not a clustering result")
    path.mkdir(parents=True, exist_ok=True)

    # meta goes first and comes back last: without it, the directory is no result
    (path / META).unlink(missing_ok=True)
    temporary = path / f"{POINTS}.tmp"
    pq.write_table(_table(clustering), temporary)
    os.replace(temporary, path / POINTS)
    temporary = path / f"{META}.tmp"
    temporary.write_text(json.dumps({"version": VERSION, **clustering.meta}, indent=2))
    os.replace(temporary, path / META)


def read_meta(path: str | Path) -> dict:
    path = Path(path)
    if not is_clustering(path):
        raise ValueError(f"{path} is not a clustering result")
    meta = json.loads((path / META).read_text())
    if meta.get("version") != VERSION:
        raise ValueError(
            f"{path}: unsupported clustering version {meta.get('version')}"
        )
    return meta


def read_labels(path: str | Path) -> np.ndarray:
    """Only the labels, without reading the coordinates."""
    return pq.read_table(Path(path) / POINTS, columns=["label"])["label"].to_numpy()


def load(path: str | Path) -> Clustering:
    meta = read_meta(path)
    table = pq.read_table(Path(path) / POINTS)

    def columns(*names: str) -> np.ndarray:
        return np.column_stack([table[name].to_numpy() for name in names])

    reduced = table["reduced"].combine_chunks()
    return Clustering(
        refs=table["ref"].to_pylist(),
        labels=table["label"].to_numpy(),
        probabilities=table["probability"].to_numpy(),
        outlier_scores=table["outlier_score"].to_numpy(),
        coords2d=columns("x2", "y2"),
        coords3d=columns("x3", "y3", "z3"),
        reduced=reduced.flatten()
        .to_numpy()
        .reshape(len(table), reduced.type.list_size),
        meta=meta,
    )


def status(meta: dict, store: Store) -> str:
    """Whether a result still matches `store`: up to date, incomplete or stale."""
    rows = meta["rows"]
    if store.model_id != meta["model_id"]:
        return f"stale: the store holds {store.model_id} embeddings now"
    if meta["edits"] is not None and meta["edits"] != store.edits:
        return "stale: rows were changed in place or removed since"
    if (
        len(store) < rows
        or fingerprint(store.refs()[:rows], meta["edits"]) != meta["refs_hash"]
    ):
        return "stale: the store holds other images in these rows"
    if len(store) > rows:
        return f"incomplete: {len(store) - rows:,} rows added since"
    return "up to date"


def _table(clustering: Clustering) -> pa.Table:
    def column(values: np.ndarray, dtype=np.float32) -> pa.Array:
        return pa.array(np.ascontiguousarray(values, dtype=dtype))

    reduced = np.ascontiguousarray(clustering.reduced, dtype=np.float32)
    coords2d, coords3d = clustering.coords2d, clustering.coords3d
    return pa.table(
        {
            "row": column(np.arange(len(clustering.refs)), np.int64),
            "ref": pa.array(clustering.refs, pa.string()),
            "label": column(clustering.labels, np.int32),
            "probability": column(clustering.probabilities),
            "outlier_score": column(clustering.outlier_scores),
            "x2": column(coords2d[:, 0]),
            "y2": column(coords2d[:, 1]),
            "x3": column(coords3d[:, 0]),
            "y3": column(coords3d[:, 1]),
            "z3": column(coords3d[:, 2]),
            "reduced": pa.FixedSizeListArray.from_arrays(
                pa.array(reduced.ravel()), reduced.shape[1]
            ),
        }
    )
