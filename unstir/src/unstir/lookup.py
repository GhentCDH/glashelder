"""`unstir lookup`: a web page that finds the building in a photo someone uploads.

The photo is matched against each collection (the reference photos and the glass
plates) in the same way as `unstir match`: DINOv2 retrieval gives the `top_k`
nearest images, ALIKED + LightGlue and a fundamental matrix score each of them,
and `decide.decide` says if the best one is a match.
"""

import io
import json
import logging
import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

from common import images
from common.fs import atomic_write

from .decide import Rule, decide, provenance, ratio
from .features import Features, location
from .match import Matcher, Settings, geometry, grayscale, points, prepare
from .poses import Pair, View, focal_35mm, scene
from .poses import relative as pose
from .store import relative

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
    features: Features | None = None
    feature_rows: dict[int, int] = field(default_factory=dict)
    """Row in `refs` to row in `features`, for the images that have them."""
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


VERDICTS = ("same", "different")
KEEP = 50
"""Recent uploads kept in memory, so feedback on them can store the photo."""
STORED_SIDE = 2048


class Feedback:
    """What visitors say about the candidates: `same` or `different` building.

    Not verified, so it is kept apart from the labels of `unstir serve`:
    `feedback.jsonl` (one line per click, the last click for a candidate counts)
    and `photos/<query>.jpg`. A photo is only written when someone gives
    feedback on it, scaled down and without its EXIF data (which can hold the
    place it was taken).
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.recent: OrderedDict[str, tuple[bytes, dict]] = OrderedDict()

    def remember(self, picture: Image.Image, result: dict) -> str:
        small = picture.copy()
        small.thumbnail((STORED_SIDE, STORED_SIDE))
        buffer = io.BytesIO()
        small.save(buffer, "JPEG", quality=90)
        query = secrets.token_hex(8)
        with self.lock:
            self.recent[query] = (buffer.getvalue(), result)
            while len(self.recent) > KEEP:
                self.recent.popitem(last=False)
        return query

    def give(
        self, lookup: "Lookup", query: str, key: str, row: int, verdict: str
    ) -> str | None:
        """Store one click. The reason it was refused, or None."""
        if verdict not in VERDICTS:
            return f"a verdict is one of {', '.join(VERDICTS)}"
        with self.lock:
            if query not in self.recent:
                return "this photo is too old, upload it again"
            photo, result = self.recent[query]
            collection = next(
                (c for c in result["collections"] if c["key"] == key), None
            )
            candidate = next(
                (c for c in (collection or {}).get("top", []) if c["row"] == row),
                None,
            )
            if collection is None or candidate is None:
                return "that image was not among the results"
            source = lookup.collections[key]
            (self.folder / "photos").mkdir(parents=True, exist_ok=True)
            stored = self.folder / "photos" / f"{query}.jpg"
            if not stored.exists():
                atomic_write(stored, photo)
            line = {
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "query": query,
                "collection": key,
                "reference": relative(source.refs[row], source.source.images),
                "verdict": verdict,
                "page_said_match": candidate["is_match"],
                "collection_match": collection["match"],
                **{
                    name: candidate[name]
                    for name in (
                        "rank",
                        "similarity",
                        "matches",
                        "f_inliers",
                        "h_inliers",
                        "cells",
                        "ratio",
                    )
                },
            }
            with (self.folder / "feedback.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(line) + "\n")
        return None


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
        store: Path | None = None,
    ):
        """`store` is a match store with the features of `unstir precompute`."""
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
            refs = data["images"]
            features = (
                Features.find(location(store, source.key), settings)
                if store is not None
                else None
            )
            self.collections[source.key] = Collection(
                source=source,
                refs=refs,
                vectors=place(data["embeddings"], self.device, label=source.title),
                features=features,
                feature_rows={}
                if features is None
                else {
                    row: features.rows[key]
                    for row, ref in enumerate(refs)
                    if (key := relative(ref, source.images)) in features.rows
                },
                known=provenance(source.images),
            )
            logger.info(f"{source.title}: {len(data['images']):,} images")
        if len(models) != 1:
            raise ValueError(f"the collections use different models: {models}")

        self.embedder = get_embedder(data["model"])
        self.embedder.load(self.device)
        self.matcher = Matcher(settings, self.device)
        logger.info(f"models loaded on {self.device}")

    def query(self, picture: Image.Image, focal: float | None = None) -> dict:
        """`focal` is the 35 mm equivalent focal length of the photo, from EXIF."""
        from embed.search import topk

        start = time.perf_counter()
        with self.lock:
            vector = self.embedder.encode_pil_image(picture).reshape(1, -1)
            image = grayscale(picture, self.settings.resize)
            features = self.matcher.extract(image)
            focal_px = focal_35mm(focal, image.shape)
            found = [
                self.match(collection, vector, image, features, topk, focal_px)
                for collection in self.collections.values()
            ]
        pairs = [pair for _, matched in found for pair in matched]
        return {
            "collections": [result for result, _ in found],
            "poses": scene(pairs, image.shape, focal_px) if pairs else None,
            "seconds": round(time.perf_counter() - start, 1),
        }

    def match(
        self, collection: Collection, vector, image, features, topk, focal: float
    ) -> tuple[dict, list[Pair]]:
        """The result for one collection, and the poses of its matched images."""
        similarities, indices = topk(vector, collection.vectors, self.top_k)
        rows = indices[0].tolist()
        stored = collection.feature_rows
        unread = [row for row in rows if row not in stored]
        candidates = dict(
            zip(
                unread,
                self.pool.map(lambda row: self.read(collection.refs[row]), unread),
                strict=True,
            )
        )
        scored: list[dict[str, Any]] = []
        kept: dict[int, tuple] = {}
        for rank, (row, similarity) in enumerate(
            zip(rows, similarities[0].tolist(), strict=True)
        ):
            if collection.features is not None and row in stored:
                theirs, shape = collection.features.get(stored[row], self.device)
            elif (picture := candidates[row]) is not None:
                theirs, shape = self.matcher.extract(picture), picture.shape
            else:
                continue
            found = self.matcher.pairs(features, theirs)
            pts, pts_theirs = points(features, found[:, 0]), points(theirs, found[:, 1])
            kept[row] = (pts, pts_theirs, found[:, 0], shape)
            scores = geometry(pts, pts_theirs, image.shape, shape, self.settings)
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
        poses = [
            found
            for reference in matches
            if (found := self.pose(collection, int(reference), kept, image, focal))
        ]
        result = {
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
        return result, poses

    def pose(
        self, collection: Collection, row: int, kept: dict, image, focal: float
    ) -> Pair | None:
        pts, pts_theirs, keypoints, shape = kept[row]
        known = collection.describe(row)
        view = View(
            key=collection.source.key,
            row=row,
            name=known.get("name") or known["file"],
            collection=collection.source.title,
        )
        return pose(
            view,
            pts,
            pts_theirs,
            keypoints,
            image.shape,
            shape,
            focal,
            self.settings.f_px,
        )

    def read(self, ref: str) -> np.ndarray | None:
        try:
            return prepare(ref, self.settings.resize)
        except OSError as error:
            logger.warning(f"could not read {ref}: {error}")
            return None


EXIF_IFD, FOCAL_35MM = 0x8769, 0xA405


def upload(data: bytes) -> tuple[Image.Image, float | None]:
    """An uploaded photo, turned upright as its EXIF data says (phone cameras
    often store the pixels sideways), in RGB. And its 35 mm equivalent focal
    length, when the EXIF data has one."""
    picture = Image.open(io.BytesIO(data))
    focal = picture.getexif().get_ifd(EXIF_IFD).get(FOCAL_35MM)
    return ImageOps.exif_transpose(picture).convert("RGB"), float(
        focal
    ) if focal else None


def create_app(lookup: Lookup, feedback: Feedback) -> Flask:
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
            picture, focal = upload(file.read())
        except (UnidentifiedImageError, OSError) as error:
            return {"error": f"that is not a photo this page can read ({error})"}, 400
        try:
            result = lookup.query(picture, focal)
        except Exception as error:
            logger.exception("lookup failed")
            return {"error": f"matching failed: {error}"}, 500
        return {**result, "query": feedback.remember(picture, result)}

    @app.route("/feedback", methods=["POST"])
    def give_feedback():
        body = request.get_json(silent=True) or {}
        try:
            row = int(str(body.get("row")))
        except ValueError:
            return {"error": "row must be a number"}, 400
        refused = feedback.give(
            lookup,
            str(body.get("query", "")),
            str(body.get("collection", "")),
            row,
            str(body.get("verdict", "")),
        )
        if refused:
            return {"error": refused}, 400
        return {"ok": True}

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
