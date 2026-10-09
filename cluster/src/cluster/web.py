"""The cluster viewer: a point cloud of a clustering result, with names and groups."""

import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, Response, render_template, request

from common import require_auth
from common.paths import rebase_paths

from . import result
from .result import Clustering
from .thumbs import samples, thumbnail

SAMPLES = 24
SIZES = range(64, 513)

logger = logging.getLogger(__name__)


def statistics(clustering: Clustering, label: int, size: int) -> dict:
    """Cheap numbers that describe one cluster, for the viewer's detail pane."""
    members = clustering.labels == label
    reduced = clustering.reduced[members]
    persistence = clustering.meta.get("persistence", [])
    return {
        "share": 100 * size / len(clustering.labels),
        "persistence": persistence[label] if label < len(persistence) else None,
        "probability": float(clustering.probabilities[members].mean()),
        "outlier": float(np.median(clustering.outlier_scores[members])),
        "spread": float(np.linalg.norm(reduced - reduced.mean(0), axis=1).mean()),
    }


def create_app(clusters: Path, image_dir: str | None, *, auth: bool) -> Flask:
    clustering = result.load(clusters)
    meta = clustering.meta
    root = image_dir if image_dir is not None else meta["root"]
    images = rebase_paths(clustering.refs, root)
    annotations = result.read_annotations(clusters)
    names, groups = annotations.names, annotations.groups
    sizes = Counter(clustering.labels.tolist())
    noise = sizes.pop(-1, 0)
    details = [
        {
            "id": label,
            "size": size,
            "samples": samples(clustering, label, SAMPLES),
            "stats": statistics(clustering, label, size),
        }
        for label, size in sizes.most_common()
    ]
    logger.info(
        f"Serving {len(images):,} points in {len(sizes):,} clusters from {clusters}"
    )

    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")
    if auth:
        require_auth(app)

    @lru_cache(maxsize=4096)
    def cached_thumbnail(row: int, size: int) -> bytes:
        return thumbnail(images[row], size)

    def binary(array: np.ndarray, dtype: str) -> Response:
        return Response(
            np.ascontiguousarray(array, dtype=dtype).tobytes(),
            mimetype="application/octet-stream",
        )

    def save() -> None:
        result.write_annotations(clusters, annotations)

    @app.route("/")
    def index():
        return render_template("viewer.html", name=Path(clusters).name, meta=meta)

    @app.route("/points")
    def points():
        return binary(clustering.coords3d, "<f4")

    @app.route("/labels")
    def labels():
        return binary(clustering.labels, "<i4")

    # only the strongest cluster of each point: that is all the viewer needs
    @app.route("/soft-labels")
    def soft_labels():
        if clustering.soft_labels is None:
            return "This result has no soft clustering", 404
        return binary(clustering.soft_labels[:, 0], "<i4")

    @app.route("/soft-scores")
    def soft_scores():
        if clustering.soft_scores is None:
            return "This result has no soft clustering", 404
        return binary(clustering.soft_scores[:, 0], "<f4")

    def soft_json() -> dict:
        soft = annotations.soft
        return {"threshold": soft.threshold, "clusters": soft.clusters}

    @app.route("/clusters")
    def cluster_list():
        return {
            "noise": noise,
            "clusters": [
                {**detail, "name": names.get(detail["id"], "")} for detail in details
            ],
            "groups": groups,
            "has_soft": clustering.soft_labels is not None,
            "soft": soft_json(),
        }

    @app.route("/soft", methods=["PUT"])
    def put_soft():
        body = request.get_json(silent=True) or {}
        threshold = body.get("threshold")
        try:
            own = {int(k): float(v) for k, v in (body.get("clusters") or {}).items()}
        except (TypeError, ValueError):
            return "Thresholds must be numbers", 400
        values = [*own.values(), *([threshold] if threshold is not None else [])]
        if not all(isinstance(v, int | float) and 0 <= v <= 1 for v in values):
            return "Thresholds must be between 0 and 1", 400
        if any(label not in sizes for label in own):
            return "Unknown cluster ids", 400
        annotations.soft = result.SoftThresholds(
            threshold=float(threshold) if threshold is not None else None,
            clusters=own,
        )
        save()
        return soft_json()

    @app.route("/thumb/<int:row>")
    def thumb(row: int):
        size = request.args.get("size", 256, type=int)
        if not 0 <= row < len(images) or size not in SIZES:
            return "No such image", 404
        try:
            jpeg = cached_thumbnail(row, size)
        except OSError as e:
            logger.warning(f"Could not read {images[row]}: {e}")
            return f"Could not read the image ({e})", 502
        response = Response(jpeg, mimetype="image/jpeg")
        response.cache_control.max_age = 86400
        return response

    @app.route("/clusters/<int:label>/name", methods=["PUT"])
    def rename(label: int):
        if label not in sizes:
            return "No such cluster", 404
        name = str((request.get_json(silent=True) or {}).get("name", "")).strip()
        if name:
            names[label] = name
        else:
            names.pop(label, None)
        save()
        return {"id": label, "name": name}

    @app.route("/groups/<path:name>", methods=["PUT"])
    def put_group(name: str):
        """Create a group, set its clusters, or rename it with `"name"`."""
        body = request.get_json(silent=True) or {}
        members = body.get("clusters", groups.get(name, []))
        new = str(body.get("name", name)).strip()
        if not new:
            return "A group needs a name", 400
        if not isinstance(members, list) or any(m not in sizes for m in members):
            return "Unknown cluster ids", 400
        if new != name and new in groups:
            return f"There is already a group {new!r}", 409
        groups.pop(name, None)
        groups[new] = sorted(set(members))
        save()
        return {"name": new, "clusters": groups[new]}

    @app.route("/groups/<path:name>", methods=["DELETE"])
    def delete_group(name: str):
        if groups.pop(name, None) is None:
            return "No such group", 404
        save()
        return {"name": name}

    return app
