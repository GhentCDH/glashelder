"""A page to label pairs as `match` or `no_match`, one pair at a time."""

import hashlib
import io
import logging
import random
from functools import lru_cache
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_file

from common import images
from common.images import is_url

from .labels import LABELS, Labels, Pair, read_pairs, resolve

SIZES = range(128, 4097)

logger = logging.getLogger(__name__)


def version(ref: str) -> str:
    """Part of every image URL, so a URL names one image, also across runs on
    the same address. Without it the browser cache can show the image of
    another pairs file at the same index."""
    return hashlib.sha1(ref.encode()).hexdigest()[:12]


def jpeg(ref: str, size: int) -> bytes:
    buffer = io.BytesIO()
    images.thumbnail(ref, size).save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def create_app(
    pairs_path: Path, labels_path: Path, *, shuffle: bool, seed: int, show_info: bool
) -> Flask:
    pairs = read_pairs(pairs_path)
    if shuffle:
        random.Random(seed).shuffle(pairs)
    labels = Labels(labels_path)
    base = pairs_path.parent
    logger.info(
        f"{len(pairs):,} pairs from {pairs_path}, "
        f"{sum(labels.get(p) is not None for p in pairs):,} labelled in {labels_path}"
    )

    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

    def find(index: int, side: str) -> str | None:
        if not 0 <= index < len(pairs) or side not in ("plate", "reference"):
            return None
        pair: Pair = pairs[index]
        return resolve(getattr(pair, side), base)

    @lru_cache(maxsize=256)
    def cached(ref: str, size: int) -> bytes:
        return jpeg(ref, size)

    @app.route("/")
    def index():
        return render_template(
            "label.html", name=pairs_path.name, labels_file=str(labels_path)
        )

    @app.route("/pairs")
    def pair_list():
        return {
            "pairs": [
                {
                    "plate": Path(p.plate).name,
                    "reference": Path(p.reference).name,
                    "info": p.info if show_info else {},
                    "label": labels.get(p),
                    "versions": {
                        "plate": version(p.plate),
                        "reference": version(p.reference),
                    },
                }
                for p in pairs
            ]
        }

    @app.route("/image/<side>/<int:index>/<tag>")
    def image(side: str, index: int, tag: str):
        ref = find(index, side)
        size = request.args.get("size", 1600, type=int)
        if (
            ref is None
            or size not in SIZES
            or tag != version(getattr(pairs[index], side))
        ):
            return "No such image", 404
        try:
            data = cached(ref, size)
        except OSError as error:
            logger.warning(f"Could not read {ref}: {error}")
            return f"Could not read the image ({error})", 502
        response = Response(data, mimetype="image/jpeg")
        response.cache_control.max_age = 86400
        return response

    @app.route("/full/<side>/<int:index>/<tag>")
    def full(side: str, index: int, tag: str):
        ref = find(index, side)
        if ref is None or tag != version(getattr(pairs[index], side)):
            return "No such image", 404
        if not (is_url(ref) or Path(ref).is_file()):
            return "No such image", 404
        return redirect(ref) if is_url(ref) else send_file(ref)

    @app.route("/label/<int:index>", methods=["PUT"])
    def label(index: int):
        if not 0 <= index < len(pairs):
            return "No such pair", 404
        value = (request.get_json(silent=True) or {}).get("label")
        if value is not None and value not in LABELS:
            return f"A label is one of {', '.join(LABELS)}, or null", 400
        labels.give(pairs[index], value)
        return {"label": value}

    return app
