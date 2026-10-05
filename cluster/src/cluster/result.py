"""A clustering of an embedding store, saved so another machine can use it.

A result is a directory:

    meta.json       what was clustered (the embedding store, its row count, its
                    `edits` and a fingerprint of its refs) and with which parameters
    points.parquet  one row per embedding row: ref, label, scores and coordinates
    annotations.json  optional: cluster names, groups and soft thresholds, from
                      `cluster serve`
    report.pdf      optional: from `cluster report`

A new clustering in the same directory drops the report and moves the annotations
aside (to `annotations-<created>.json`), because both refer to the old cluster ids.

Parquet, so anything reads it (polars, pandas, DuckDB, a browser) without this
package. Refs are kept as the embedding store has them: relative to its image
root, or URLs.

A result covers the first `rows` rows of its store. Rows appended later do not
change those, so the result stays right for them and only misses the new ones.
Rows changed in place or removed (the store's `edits`) make it stale.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embed.neighbours import queries_hash
from embed.store import Store

META, POINTS, REPORT = "meta.json", "points.parquet", "report.pdf"
ANNOTATIONS = "annotations.json"
VERSION = 1

logger = logging.getLogger(__name__)


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
    soft_labels: np.ndarray | None = None
    """`(n, top)` the strongest clusters of each point by soft clustering, best
    first, -1 where there are fewer clusters. None when it was not computed."""
    soft_scores: np.ndarray | None = None
    """`(n, top)` the membership score of each of those clusters."""


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

    if is_clustering(path):
        # annotations and the report belong to the old cluster ids
        old = read_meta(path)
        if (path / ANNOTATIONS).is_file():
            stamp = old["created"].replace(":", "")
            os.replace(path / ANNOTATIONS, path / f"annotations-{stamp}.json")
        (path / REPORT).unlink(missing_ok=True)

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

    def matrix(name: str) -> np.ndarray | None:
        if name not in table.column_names:
            return None
        column = table[name].combine_chunks()
        return column.flatten().to_numpy().reshape(len(table), column.type.list_size)

    reduced = matrix("reduced")
    assert reduced is not None
    return Clustering(
        refs=table["ref"].to_pylist(),
        labels=table["label"].to_numpy(),
        probabilities=table["probability"].to_numpy(),
        outlier_scores=table["outlier_score"].to_numpy(),
        coords2d=columns("x2", "y2"),
        coords3d=columns("x3", "y3", "z3"),
        reduced=reduced,
        meta=meta,
        soft_labels=matrix("soft_labels"),
        soft_scores=matrix("soft_scores"),
    )


@dataclass
class SoftThresholds:
    """When a noise point joins its strongest cluster by soft clustering."""

    threshold: float | None = None
    """For every cluster without its own: the score a point needs. None: none join."""
    clusters: dict[int, float] = field(default_factory=dict)
    """Thresholds of single clusters, by cluster id."""

    def of(self, label: int) -> float | None:
        return self.clusters.get(label, self.threshold)


@dataclass
class Annotations:
    """What people added to a clustering by hand."""

    names: dict[int, str] = field(default_factory=dict)
    """Human names of clusters, by cluster id."""
    groups: dict[str, list[int]] = field(default_factory=dict)
    """Named groups of cluster ids. A cluster can be in more than one group."""
    soft: SoftThresholds = field(default_factory=SoftThresholds)


def assign(clustering: Clustering, soft: SoftThresholds) -> np.ndarray:
    """The labels, with noise points moved into their strongest cluster where its
    score reaches that cluster's threshold. Points already in a cluster stay there."""
    labels = clustering.labels.copy()
    if clustering.soft_labels is None or clustering.soft_scores is None:
        return labels
    strongest = clustering.soft_labels[:, 0]
    scores = clustering.soft_scores[:, 0]
    # a threshold per cluster id, inf where there is none
    count = int(max(labels.max(), strongest.max())) + 1
    thresholds = np.array(
        [t if (t := soft.of(label)) is not None else np.inf for label in range(count)]
    )
    joins = (labels == -1) & (strongest >= 0)
    joins[joins] = scores[joins] >= thresholds[strongest[joins]]
    labels[joins] = strongest[joins]
    return labels


def identity(meta: dict) -> str:
    """Which clustering annotations belong to: a new run of the same data differs."""
    return f"{meta['refs_hash']}:{meta['created']}"


def read_annotations(path: str | Path) -> Annotations:
    path = Path(path)
    if not (path / ANNOTATIONS).is_file():
        return Annotations()
    saved = json.loads((path / ANNOTATIONS).read_text(encoding="utf-8"))
    if saved.get("clustering") != identity(read_meta(path)):
        logger.warning(
            f"{path / ANNOTATIONS} belongs to another clustering, ignoring it"
        )
        return Annotations()
    soft = saved.get("soft", {})
    return Annotations(
        names={int(label): name for label, name in saved.get("names", {}).items()},
        groups=saved.get("groups", {}),
        soft=SoftThresholds(
            threshold=soft.get("threshold"),
            clusters={int(k): v for k, v in soft.get("clusters", {}).items()},
        ),
    )


def write_annotations(path: str | Path, annotations: Annotations) -> None:
    path = Path(path)
    names, groups, soft = annotations.names, annotations.groups, annotations.soft
    temporary = path / f"{ANNOTATIONS}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "clustering": identity(read_meta(path)),
                "names": {str(label): names[label] for label in sorted(names)},
                "groups": {name: sorted(groups[name]) for name in sorted(groups)},
                "soft": {
                    "threshold": soft.threshold,
                    "clusters": {
                        str(k): soft.clusters[k] for k in sorted(soft.clusters)
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path / ANNOTATIONS)


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

    def matrix(values: np.ndarray, dtype=np.float32) -> pa.Array:
        values = np.ascontiguousarray(values, dtype=dtype)
        return pa.FixedSizeListArray.from_arrays(
            pa.array(values.ravel()), values.shape[1]
        )

    coords2d, coords3d = clustering.coords2d, clustering.coords3d
    columns = {
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
        "reduced": matrix(clustering.reduced),
    }
    if clustering.soft_labels is not None and clustering.soft_scores is not None:
        columns["soft_labels"] = matrix(clustering.soft_labels, np.int32)
        columns["soft_scores"] = matrix(clustering.soft_scores)
    return pa.table(columns)
