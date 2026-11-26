"""The master file: every checked photo with its metadata and its label.

`verify` writes it on every save, `archstil master` rebuilds it. It joins three
sources by image stem (the Beeldbank image id):

- `references.jsonl` of `scrape`: the object, its styles, dating, type, licence;
- the `<stem>.json` of `segment`: image size and the SAM3 segments;
- `selections.json` of `verify`: the chosen segments, class and annotation.

Per verified photo the annotation is the union of the chosen segments: the object
as one instance, with a box and its outline as polygons in pixels.
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
"""Polygons smaller than this fraction of the largest one are dropped as specks."""
SIMPLIFY = 0.002
"""Douglas-Peucker tolerance, as a fraction of the outline's length."""


def annotation(masks: list[np.ndarray]) -> dict | None:
    """Box and polygons of the union of `masks`, in pixels; `None` when empty."""
    union = np.logical_or.reduce(masks).astype(np.uint8)
    if not union.any():
        return None
    ys, xs = np.nonzero(union)
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: float(cv2.contourArea(c)), reverse=True)
    largest = cv2.contourArea(contours[0]) if contours else 0
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_PART * largest:
            break
        simple = cv2.approxPolyDP(
            contour, SIMPLIFY * cv2.arcLength(contour, closed=True), closed=True
        )
        if len(simple) >= 3:
            polygons.append(simple.reshape(-1).astype(int).tolist())
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
