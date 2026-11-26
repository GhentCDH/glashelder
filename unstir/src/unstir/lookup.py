"""`unstir lookup`: a web page that finds the building in a photo someone uploads.

The photo is matched against each collection (the reference photos and the glass
plates) in the same way as `unstir match`: DINOv2 retrieval gives the `top_k`
nearest images, ALIKED + LightGlue and a fundamental matrix score each of them,
and `decide.decide` says if the best one is a match.
"""

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

from common import images

from .decide import Rule, decide, provenance, ratio
from .match import Matcher, Settings, geometry, grayscale, prepare

THREADS = 16
SIZES = range(64, 2049)
MAX_UPLOAD = 30 * 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """One collection to match against, as given on the command line."""

    key: str
    title: str
    embeddings: Path
    images: Path
    many: bool = False
    """The collection can hold several images of one building (the plates).
    Then every candidate that passes `min_f` and `min_ratio` is a match, and the
    margin is not used: a second plate of the same church is no look-alike."""


@dataclass
class Collection:
    source: Source
    refs: list[str]
    vectors: object
    """The embeddings, placed where searching them is fastest."""
    known: dict[str, dict] = field(default_factory=dict)
    """Names, places and URIs by file name, from `provenance.jsonl`."""

    def describe(self, row: int) -> dict:
        name = Path(self.refs[row]).name
        return {"file": name, **self.known.get(name, {})}


def passes(candidate: dict, rule: Rule) -> bool:
    return candidate["f_inliers"] >= rule.min_f and candidate["ratio"] >= rule.min_ratio


def reason(best: dict, rule: Rule, *, many: int | None = None) -> str:
    """Why the best candidate is, or is not, a match, in words. `many` is the
    number of matches in a collection that can hold several per building."""
    if many:
        images = "image shows" if many == 1 else "images show"
        return (
            f"{many} {images} this building: at least {rule.min_f} matched points "
            f"and {rule.min_ratio:.0%} of them fit one 3D geometry"
        )
    if best["decision"] == "match":
        return (
            f"{best['f_inliers']} matched points fit one 3D geometry, "
            f"{best['margin']} more than the next candidate"
        )
    if best["f_inliers"] < rule.min_f:
        return (
            f"the best candidate has {best['f_inliers']} matched points that fit, "
            f"a match needs {rule.min_f}"
        )
    if best["ratio"] < rule.min_ratio:
        return (
            f"only {best['ratio']:.0%} of the matched points fit one geometry, "
            f"a match needs {rule.min_ratio:.0%}"
        )
    return (
        f"the best candidate is only {best['margin']} points ahead of the next, "
        f"a match needs {rule.margin}: two buildings look alike"
    )


class Lookup:
    """The models and collections, loaded once. One query at a time."""

    def __init__(
        self,
        sources: list[Source],
        *,
        top_k: int,
        show: int,
        rule: Rule,
        settings: Settings,
    ):
        from embed.device import get_device
        from embed.io import load_embeddings
        from embed.models import get_embedder
        from embed.search import place

        self.device = get_device()
        self.top_k, self.show, self.rule, self.settings = top_k, show, rule, settings
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(THREADS)

        self.collections: dict[str, Collection] = {}
        models = set()
        for source in sources:
            data = load_embeddings(source.embeddings, root=source.images)
            models.add(data["model_id"])
            self.collections[source.key] = Collection(
                source=source,
                refs=data["images"],
                vectors=place(data["embeddings"], self.device, label=source.title),
                known=provenance(source.images),
            )
            logger.info(f"{source.title}: {len(data['images']):,} images")
        if len(models) != 1:
            raise ValueError(f"the collections use different models: {models}")

        self.embedder = get_embedder(data["model"])
        self.embedder.load(self.device)
        self.matcher = Matcher(settings, self.device)
        logger.info(f"models loaded on {self.device}")

    def query(self, picture: Image.Image) -> dict:
        from embed.search import topk

        start = time.perf_counter()
        with self.lock:
            vector = self.embedder.encode_pil_image(picture).reshape(1, -1)
            image = grayscale(picture, self.settings.resize)
            features = self.matcher.extract(image)
            results = [
                self.match(collection, vector, image, features, topk)
                for collection in self.collections.values()
            ]
        return {
            "collections": results,
            "seconds": round(time.perf_counter() - start, 1),
        }

    def match(self, collection: Collection, vector, image, features, topk) -> dict:
        similarities, indices = topk(vector, collection.vectors, self.top_k)
        rows = indices[0].tolist()
        candidates = dict(
            zip(
                rows,
                self.pool.map(lambda row: self.read(collection.refs[row]), rows),
                strict=True,
            )
        )
        scored: list[dict[str, Any]] = []
        for rank, (row, similarity) in enumerate(
            zip(rows, similarities[0].tolist(), strict=True)
        ):
            theirs = candidates[row]
            if theirs is None:
                continue
            pts, pts_theirs = self.matcher.match(features, self.matcher.extract(theirs))
            scores = geometry(pts, pts_theirs, image.shape, theirs.shape, self.settings)
            scored.append(
                {
                    "plate": "query",
                    "reference": str(row),
                    "row": row,
                    "rank": rank,
                    "similarity": round(float(similarity), 3),
                    **scores,
                    "ratio": round(ratio(scores), 3),
                }
            )

        top = sorted(scored, key=lambda s: -s["f_inliers"])[: self.show]
        if not scored:
            matches, explained = set(), "no candidate could be read"
        elif collection.source.many:
            matches = {s["reference"] for s in scored if passes(s, self.rule)}
            best = decide(scored, self.rule)[0]
            explained = reason(best, self.rule, many=len(matches))
        else:
            best = decide(scored, self.rule)[0]
            matches = {best["reference"]} if best["decision"] == "match" else set()
            explained = reason(best, self.rule)
        return {
            "key": collection.source.key,
            "title": collection.source.title,
            "match": bool(matches),
            "matches": len(matches),
            "reason": explained,
            "top": [
                {
                    **s,
                    **collection.describe(s["row"]),
                    "is_match": s["reference"] in matches,
                }
                for s in top
            ],
        }

    def read(self, ref: str) -> np.ndarray | None:
        try:
            return prepare(ref, self.settings.resize)
        except OSError as error:
            logger.warning(f"could not read {ref}: {error}")
            return None


def upload(data: bytes) -> Image.Image:
    """An uploaded photo, turned upright as its EXIF data says (phone cameras
    often store the pixels sideways), in RGB."""
    picture = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(picture).convert("RGB")


def create_app(lookup: Lookup) -> Flask:
    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

    @lru_cache(maxsize=1024)
    def thumbnail(ref: str, size: int) -> bytes:
        buffer = io.BytesIO()
        images.thumbnail(ref, size).save(buffer, "JPEG", quality=85)
        return buffer.getvalue()

    def find(key: str, row: int) -> str | None:
        collection = lookup.collections.get(key)
        if collection is None or not 0 <= row < len(collection.refs):
            return None
        return collection.refs[row]

    @app.route("/")
    def index():
        return render_template(
            "lookup.html",
            collections=[c.source.title for c in lookup.collections.values()],
            top_k=lookup.top_k,
        )

    @app.route("/lookup", methods=["POST"])
    def run_lookup():
        file = request.files.get("image")
        if file is None or not file.filename:
            return {"error": "no photo was sent"}, 400
        try:
            picture = upload(file.read())
        except (UnidentifiedImageError, OSError) as error:
            return {"error": f"that is not a photo this page can read ({error})"}, 400
        try:
            return lookup.query(picture)
        except Exception as error:
            logger.exception("lookup failed")
            return {"error": f"matching failed: {error}"}, 500

    @app.route("/thumb/<key>/<int:row>")
    def thumb(key: str, row: int):
        ref = find(key, row)
        size = request.args.get("size", 400, type=int)
        if ref is None or size not in SIZES:
            return "No such image", 404
        try:
            data = thumbnail(ref, size)
        except OSError as error:
            return f"Could not read the image ({error})", 502
        response = Response(data, mimetype="image/jpeg")
        response.cache_control.max_age = 86400
        return response

    @app.route("/full/<key>/<int:row>")
    def full(key: str, row: int):
        ref = find(key, row)
        if ref is None or not Path(ref).is_file():
            return "No such image", 404
        return send_file(ref)

    @app.errorhandler(413)
    def too_large(_):
        return {"error": f"the photo is larger than {MAX_UPLOAD // 2**20} MB"}, 413

    return app
