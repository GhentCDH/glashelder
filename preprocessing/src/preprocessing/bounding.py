from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int


def load_bounding_boxes(path: Path) -> dict[str, BoundingBox]:
    """Load a bounding-box JSON into {stem: BoundingBox}."""
    import json

    with open(path) as f:
        raw = json.load(f)
    return {stem: BoundingBox(**box) for stem, box in raw.items()}


def save_bounding_boxes(boxes: dict[str, BoundingBox], path: Path) -> None:
    """Dump {stem: BoundingBox} to JSON."""
    import json

    with open(path, "w") as f:
        json.dump(
            {stem: asdict(box) for stem, box in boxes.items()}, f, ensure_ascii=False
        )


def _find_bounding_box(
    img_path: Path, crops_dir: Path
) -> tuple[str, BoundingBox | None, list[tuple[str, str]]]:
    """Returns (stem, box_or_none, log_messages)."""
    import cv2

    messages = []

    img = cv2.imread(str(img_path))
    if img is None:
        messages.append(("error", f" {img_path.stem}: couldn't read image, skip"))
        return img_path.stem, None, messages

    crop = cv2.imread(str(crops_dir / img_path.name))
    if crop is None:
        messages.append(("error", f" {img_path.stem}: couldn't read crop, skip"))
        return img_path.stem, None, messages

    result = cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.99:
        messages.append(
            (
                "error",
                f" {img_path.stem}: no good template match (max_val={max_val:.3f}), skip",
            )
        )
        return img_path.stem, None, messages

    x, y = max_loc
    h, w = crop.shape[:2]
    messages.append(
        ("info", f"  {img_path.stem}: found bounding box {x=} {y=} {w=} {h=}")
    )
    return img_path.stem, BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)), messages
