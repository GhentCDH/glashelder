"""The verify viewer: pick which SAM3 segment is the heritage object in each photo.

Reads the output of `archstil segment` (per image `<stem>.json`, the masks and a copy
of the original) and the `references.jsonl` of `archstil scrape` for the object's
name and style. Both may still be growing: the viewer looks for new photos on every
page, in the order they were segmented, so you can label while `scrape` and
`segment --watch` run.

Choices go to `selections.json` in the segment output, one entry per photo:

    {"<stem>": {"status": "verified" | "rejected", "selected": [mask ids],
                "class": "gotiek", "annotation": {"bbox_xyxy", "polygons", ...}}}

`verified`: the selected masks are the object, `class` is its style. `rejected`: no
mask is right (wrong photo, object not visible, SAM3 missed it). After every save the
master file (see `master.py`) is written again.
"""

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image

from . import master
from .master import SELECTIONS, annotation, default_class, load_references, write_json

logger = logging.getLogger(__name__)

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
]


def colour(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


class Live:
    """Segmented photos and references, read again whenever they changed on disk."""

    def __init__(self, segments: Path, references: Path | None):
        self.segments = segments
        self.references = references
        self._stems: list[str] = []
        self._stems_key: tuple[int, float] | None = None
        self._refs: dict[str, dict] = {}
        self._refs_key: tuple[int, float] | None = None

    def stems(self) -> list[str]:
        """Segmented photos, oldest first, so new ones are added at the end."""
        key = (len(list(self.segments.iterdir())), self.segments.stat().st_mtime)
        if key != self._stems_key:
            done = [
                p
                for p in self.segments.glob("*.json")
                if p.name not in (SELECTIONS, master.MASTER)
            ]
            done.sort(key=lambda p: (p.stat().st_mtime, p.name))
            self._stems = [p.stem for p in done]
            self._stems_key = key
        return self._stems

    def refs(self) -> dict[str, dict]:
        if self.references is None or not self.references.is_file():
            return {}
        stat = self.references.stat()
        key = (stat.st_size, stat.st_mtime)
        if key != self._refs_key:
            self._refs = load_references(self.references)
            self._refs_key = key
        return self._refs


def create_app(segments: Path, references: Path | None, master_path: Path) -> Flask:
    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

    live = Live(segments, references)
    selections_path = segments / SELECTIONS
    selections: dict[str, dict] = (
        json.loads(selections_path.read_text()) if selections_path.is_file() else {}
    )
    logger.info(
        "%d segmented photos, %d references, %d already checked; master file %s",
        len(live.stems()),
        len(live.refs()),
        len(selections),
        master_path,
    )

    def segment(stem: str) -> dict:
        path = segments / f"{stem}.json"
        if not path.is_file():
            abort(404)
        return json.loads(path.read_text())

    @lru_cache(maxsize=256)
    def mask(stem: str, mask_id: int) -> np.ndarray:
        path = segments / f"{stem}_masks" / f"{mask_id}.png"
        if not path.is_file():
            abort(404)
        return np.asarray(Image.open(path).convert("L")) > 127

    def default_selection(buildings: list[dict]) -> list[int]:
        """A guess at the object: the largest segment, as photos tend to centre it."""
        if not buildings:
            return []
        return [max(buildings, key=lambda b: b["mask_area_px"])["id"]]

    def next_open(stems: list[str], after: int) -> str | None:
        """The first photo after position `after` not checked yet, wrapping around."""
        for offset in range(1, len(stems) + 1):
            stem = stems[(after + offset) % len(stems)]
            if stem not in selections:
                return stem
        return None

    def render(stem: str):
        stems = live.stems()
        if stem not in stems:
            abort(404)
        index = stems.index(stem)
        data = segment(stem)
        ref = live.refs().get(stem)
        saved = selections.get(stem)
        buildings = data["buildings"]
        for b in buildings:
            b["colour"] = "rgb({},{},{})".format(*colour(b["id"]))
        return render_template(
            "verify.html",
            index=index,
            total=len(stems),
            stem=stem,
            prev=stems[index - 1] if index > 0 else None,
            next=stems[index + 1] if index + 1 < len(stems) else None,
            next_open=next_open(stems, index),
            data=data,
            buildings=buildings,
            ref=ref,
            saved=saved,
            selected=saved["selected"] if saved else default_selection(buildings),
            removed=(saved or {}).get("removed", []),
            chosen_class=(saved or {}).get("class") or default_class(ref),
            checked=len(selections),
            verified=sum(s["status"] == "verified" for s in selections.values()),
        )

    @app.route("/")
    def index():
        stems = live.stems()
        if not stems:
            return (
                f"No `archstil segment` output in {segments} yet. "
                "Reload once the first photo is segmented.",
                404,
            )
        stem = next_open(stems, len(stems) - 1)
        return render(stem or stems[0])

    @app.route("/photo/<stem>")
    def show(stem: str):
        return render(stem)

    @app.route("/original/<stem>")
    def original(stem: str):
        return send_file(segments / segment(stem)["original"])

    @app.route("/mask/<stem>/<int:mask_id>.png")
    def mask_png(stem: str, mask_id: int):
        """The mask as a coloured, transparent overlay."""
        from io import BytesIO

        m = mask(stem, mask_id)
        rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
        rgba[m] = (*colour(mask_id), 255)
        buffer = BytesIO()
        Image.fromarray(rgba).save(buffer, format="PNG")
        buffer.seek(0)
        response = send_file(buffer, mimetype="image/png")
        response.cache_control.max_age = 3600
        return response

    @app.route("/hit/<stem>")
    def hit(stem: str):
        """Masks under a point, smallest first, so a click picks the innermost."""
        x, y = request.args.get("x", type=int), request.args.get("y", type=int)
        if x is None or y is None:
            abort(400)
        buildings = segment(stem)["buildings"]
        under = [
            b
            for b in buildings
            if (m := mask(stem, b["id"])).shape[0] > y >= 0
            and m.shape[1] > x >= 0
            and m[y, x]
        ]
        return jsonify(
            [b["id"] for b in sorted(under, key=lambda b: b["mask_area_px"])]
        )

    @app.route("/save/<stem>", methods=["POST"])
    def save(stem: str):
        body = request.get_json(force=True)
        status = body.get("status")
        if status not in ("verified", "rejected"):
            abort(400)
        known = {b["id"] for b in segment(stem)["buildings"]}
        selected = sorted(int(i) for i in body.get("selected", []) if int(i) in known)
        removed = sorted(
            int(i)
            for i in body.get("removed", [])
            if int(i) in known and int(i) not in selected
        )
        if status == "verified" and not selected:
            abort(400, "select at least one segment, or reject the photo")

        ref = live.refs().get(stem)
        labels = [s["label"] for s in (ref or {}).get("styles", [])]
        chosen = body.get("class") or default_class(ref)
        if labels and chosen not in labels:
            abort(400, f"class must be one of the object's styles: {labels}")

        entry: dict = {
            "status": status,
            "selected": selected if status == "verified" else [],
            "removed": removed if status == "verified" else [],
            "class": chosen if status == "verified" else None,
            "annotation": None,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if status == "verified":
            entry["annotation"] = annotation(
                [mask(stem, i) for i in selected], [mask(stem, i) for i in removed]
            )
            if entry["annotation"] is None:
                abort(400, "the removed segments cover all of the selected ones")
        # kept here so the master file can be rebuilt without opening every segment
        data = segment(stem)
        entry |= {k: data[k] for k in ("image", "width", "height")}
        selections[stem] = entry
        write_json(selections_path, selections)
        written = master.write(segments, selections, live.refs(), master_path)
        return jsonify(ok=True, checked=len(selections), verified=written["verified"])

    return app
