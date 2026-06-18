from pathlib import Path


def _find_bounding_box(
    img_path: Path, crops_dir: Path
) -> tuple[str, dict | None, list[tuple[str, str]]]:
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
    return img_path.stem, {"x": x, "y": y, "w": w, "h": h}, messages
