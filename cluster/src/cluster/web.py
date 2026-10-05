"""The cluster viewer: a point cloud of a clustering result, with names to give."""

import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, Response, render_template, request

from common import require_auth
from embed import rebase_paths

from . import result
from .thumbs import samples, thumbnail

SAMPLES = 12
SIZES = range(64, 513)

logger = logging.getLogger(__name__)


def create_app(clusters: Path, image_dir: str | None, *, auth: bool) -> Flask:
    clustering = result.load(clusters)
    meta = clustering.meta
    root = image_dir if image_dir is not None else meta["root"]
    images = rebase_paths(clustering.refs, root)
    names = result.read_names(clusters)
    sizes = Counter(clustering.labels.tolist())
    noise = sizes.pop(-1, 0)
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

    @app.route("/")
    def index():
        return render_template("viewer.html", name=Path(clusters).name, meta=meta)

    @app.route("/points")
    def points():
        return binary(clustering.coords3d, "<f4")

    @app.route("/labels")
    def labels():
        return binary(clustering.labels, "<i4")

    @app.route("/clusters")
    def cluster_list():
        return {
            "noise": noise,
            "clusters": [
                {
                    "id": label,
                    "size": size,
                    "name": names.get(label, ""),
                    "samples": samples(clustering, label, SAMPLES),
                }
                for label, size in sizes.most_common()
            ],
        }

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
        result.write_names(clusters, names)
        return {"id": label, "name": name}

    return app
