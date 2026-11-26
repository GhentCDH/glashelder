"""The master file: every checked photo with its metadata and its label.

`verify` writes it on every save, `archstil master` rebuilds it. It joins three
sources by image stem (the Beeldbank image id):

- `references.jsonl` of `scrape`: the object, its styles, dating, type, licence;
- the `<stem>.json` of `segment`: image size and the SAM3 segments;
- `selections.json` of `verify`: the chosen segments, class and annotation.

Per verified photo the annotation is the union of the chosen segments minus the
segments marked as removed: the object as one instance, with a box and its outline
as polygons in pixels.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

MASTER = "master.json"
SELECTIONS = "selections.json"
MIN_PART = 0.01
"""Parts smaller than this fraction of the largest one are dropped as specks."""
MIN_HOLE = 0.001
"""Holes smaller than this fraction of their part are ignored as mask noise."""
SIMPLIFY = 1.0
"""Douglas-Peucker tolerance in pixels. Fixed, not relative to the outline's length:
outlines broken up by trees are so long that a relative tolerance cuts corners."""


def simplify(contour: np.ndarray) -> np.ndarray:
    """The contour with fewer points, as an (n, 2) array."""
    return cv2.approxPolyDP(contour, SIMPLIFY, closed=True).reshape(-1, 2)


def bridge(ring: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """One ring that leaves out `holes`, joined to the outline by zero-width cuts.

    YOLO polygons (and most polygon formats) cannot have holes. Walking from the
    nearest outline point into each hole, around it and back gives the same area as
    one simple ring.
    """
    for hole in sorted(holes, key=lambda h: -cv2.contourArea(h)):
        distances = ((ring[:, None, :] - hole[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(distances.argmin()), distances.shape)
        around = np.concatenate([hole[j:], hole[: j + 1]])
        ring = np.concatenate([ring[: i + 1], around, ring[i:]])
    return ring


def annotation(
    masks: list[np.ndarray], removed: list[np.ndarray] | None = None
) -> dict | None:
    """Box and polygons of the union of `masks` minus the union of `removed`, in
    pixels; `None` when nothing is left.

    A removed segment inside the selected ones leaves a hole; it is bridged into the
    outline (see `bridge`), so every polygon is one simple ring without the hole.
    """
    union = np.logical_or.reduce(masks)
    if removed:
        union &= ~np.logical_or.reduce(removed)
    union = union.astype(np.uint8)
    if not union.any():
        return None

    # two levels: outlines of the parts, and the holes inside each of them
    contours, hierarchy = cv2.findContours(
        union, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    links = hierarchy[0]
    parts = [i for i, link in enumerate(links) if link[3] < 0]
    area = {i: float(cv2.contourArea(contours[i])) for i in range(len(contours))}
    largest = max(area[i] for i in parts)

    polygons, kept = [], np.zeros_like(union)
    for i in sorted(parts, key=lambda i: -area[i]):
        if area[i] < MIN_PART * largest:
            break
        holes = [
            simplify(contours[h])
            for h, link in enumerate(links)
            if link[3] == i and area[h] >= MIN_HOLE * area[i]
        ]
        ring = bridge(simplify(contours[i]), [h for h in holes if len(h) >= 3])
        if len(ring) >= 3:
            polygons.append(ring.reshape(-1).astype(int).tolist())
            cv2.fillPoly(kept, [ring.astype(np.int32)], 1)

    # the box of what the polygons keep, so dropped specks cannot stretch it
    ys, xs = np.nonzero(kept & union)
    if not len(xs):
        ys, xs = np.nonzero(union)
    return {
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "polygons": polygons,
        "area_px": int(union.sum()),
    }


def default_class(ref: dict | None) -> str | None:
    """The style the object was downloaded for, else its first style."""
    if not ref or not ref.get("styles"):
        return None
    wanted = (ref.get("query") or {}).get("stijl")
    for style in ref["styles"]:
        if style["uri"] == wanted:
            return style["label"]
    return ref["styles"][0]["label"]


def load_references(path: Path | None) -> dict[str, dict]:
    """References by image stem."""
    if path is None or not path.is_file():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        try:
            ref = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short while scrape is writing it
        ref["references_dir"] = str(path.parent)  # image_file is relative to it
        found[Path(ref["image_file"]).stem] = ref
    return found


def write_json(path: Path, data: object) -> None:
    """Write whole or not at all, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def entry(
    stem: str, selection: dict, segment: dict, ref: dict | None, base: Path
) -> dict:
    """One photo of the master file."""
    ref = ref or {}
    image = (
        (Path(ref["references_dir"]) / ref["image_file"])
        if "references_dir" in ref
        else Path(segment["image"])
    )
    return {
        "stem": stem,
        "image": os.path.relpath(image, base),
        "width": segment["width"],
        "height": segment["height"],
        "status": selection["status"],
        "class": selection.get("class"),
        "annotation": selection.get("annotation"),
        "segments": selection.get("selected", []),
        "removed_segments": selection.get("removed", []),
        "name": ref.get("name"),
        "location": ref.get("location"),
        "styles": [s["label"] for s in ref.get("styles", [])],
        "datings": [d["label"] for d in ref.get("datings", [])],
        "types": [t["label"] for t in ref.get("types", [])],
        "object_id": ref.get("object_id"),
        "object_uri": ref.get("object_uri"),
        "image_id": ref.get("image_id"),
        "image_uri": ref.get("image_uri"),
        "image_title": ref.get("image_title"),
        "image_date": ref.get("image_date"),
        "attribution": ref.get("attribution"),
        "licenses": ref.get("licenses", []),
        "checked_at": selection.get("checked_at"),
    }


def build(
    segments: Path, selections: dict[str, dict], refs: dict[str, dict], master: Path
) -> dict:
    """The master file's content: the classes, then every checked photo."""
    images = []
    for stem, selection in sorted(selections.items()):
        if "width" in selection:  # saved with the selection by `verify`
            segment = selection
        else:
            segment_path = segments / f"{stem}.json"
            if not segment_path.is_file():
                continue
            segment = json.loads(segment_path.read_text())
        images.append(entry(stem, selection, segment, refs.get(stem), master.parent))

    verified = [i for i in images if i["status"] == "verified" and i["annotation"]]
    classes = sorted({i["class"] for i in verified if i["class"]}, key=str.casefold)
    ids = {name: n for n, name in enumerate(classes)}
    for image in images:
        image["class_id"] = ids.get(image["class"]) if image in verified else None
    return {
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": classes,
        "counts": {name: sum(i["class"] == name for i in verified) for name in classes},
        "verified": len(verified),
        "rejected": sum(i["status"] == "rejected" for i in images),
        "images": images,
    }


def write(
    segments: Path, selections: dict[str, dict], refs: dict[str, dict], master: Path
) -> dict:
    data = build(segments, selections, refs, master)
    write_json(master, data)
    return data
