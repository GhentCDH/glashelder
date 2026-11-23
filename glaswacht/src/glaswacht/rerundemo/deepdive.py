"""Layer B: probe the three models on a few plates and keep every step.

`probe` needs the GPU and saves traces to a folder. `load_traces` needs neither, so the
demo (see `generation.py`) can be rebuilt on any machine.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from glaswacht.models.safety_model import Rating, SeenImage, TokenStep, Trace

logger = logging.getLogger(__name__)

PLATE_PX = 768


@dataclass
class Plate:
    name: str
    traces: dict[str, Trace]


@dataclass
class DeepRun:
    models: tuple[str, ...]
    plates: list[Plate]
    load_s: dict[str, float]
    plate_dir: Path


# --------------------------------------------------------------------------- probing


def probe(images: list[Path], model_classes: list, out_dir: Path) -> None:
    """Run every model over `images`, saving one trace per (plate, model)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        d = out_dir / path.stem
        d.mkdir(exist_ok=True)
        with Image.open(path) as im:
            small = im.convert("RGB")
            small.thumbnail((PLATE_PX, PLATE_PX))
            small.save(d / "plate.jpg", quality=88)

    load_s: dict[str, float] = {}
    for cls in model_classes:
        model = cls()
        t0 = time.perf_counter()
        model.load_model()
        load_s[cls.name] = time.perf_counter() - t0
        logger.info("%s loaded in %.1fs", cls.name, load_s[cls.name])
        try:
            for path in images:
                with Image.open(path) as im:
                    trace = model.inspect(im.convert("RGB"), path.name)
                save_trace(out_dir / path.stem, trace)
                logger.info("%s %s -> %s", cls.name, path.name, trace.rating.rating)
        finally:
            model.unload_model()
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "models": [c.name for c in model_classes],
                "plates": [p.stem for p in images],
                "load_s": load_s,
            },
            indent=2,
        )
    )


def save_trace(d: Path, trace: Trace) -> None:
    meta = asdict(trace)
    meta.pop("seen")
    meta["seen"] = []
    for i, s in enumerate(trace.seen):
        fname = f"{trace.model}_seen_{i}.png"
        Image.fromarray(s.pixels).save(d / fname)
        meta["seen"].append({"name": s.name, "file": fname, "patch_size": s.patch_size})
    (d / f"{trace.model}.json").write_text(json.dumps(meta))


def load_traces(root: Path) -> DeepRun:
    run = json.loads((root / "run.json").read_text())
    plates = []
    for stem in run["plates"]:
        d = root / stem
        traces = {}
        for m in run["models"]:
            meta = json.loads((d / f"{m}.json").read_text())
            seen = [
                SeenImage(
                    s["name"],
                    np.asarray(Image.open(d / s["file"]).convert("RGB")),
                    s["patch_size"],
                )
                for s in meta.pop("seen")
            ]
            meta["steps"] = [
                TokenStep(**{**s, "top": [tuple(t) for t in s["top"]]})
                for s in meta["steps"]
            ]
            meta["rating"] = Rating(**meta["rating"])
            traces[m] = Trace(seen=seen, **meta)
        plates.append(Plate(stem, traces))
    return DeepRun(tuple(run["models"]), plates, run["load_s"], root)


# --------------------------------------------------------------------------- images


def mosaic(seen: list[SeenImage], aspect: float | None = None) -> np.ndarray:
    """Tile what the model sees into one image (a 2D view would stack them).

    With `aspect` (width / height of the original), the first image is treated as an
    "anyres" overview and the rest as the crop grid it was cut into: the overview goes
    on the left, the crops in their real rows and columns on the right.
    """
    if len(seen) == 1:
        return seen[0].pixels
    if aspect is None or len(seen) < 3:
        cols = int(np.ceil(np.sqrt(len(seen))))
        h = max(s.pixels.shape[0] for s in seen)
        w = max(s.pixels.shape[1] for s in seen)
        rows = int(np.ceil(len(seen) / cols))
        canvas = np.full((rows * h, cols * w, 3), 32, np.uint8)
        for i, s in enumerate(seen):
            r, c = divmod(i, cols)
            ph, pw = s.pixels.shape[:2]
            canvas[r * h : r * h + ph, c * w : c * w + pw] = s.pixels
        return canvas

    base, crops = seen[0].pixels, [s.pixels for s in seen[1:]]
    n = len(crops)
    # the factorisation rows x cols whose shape best matches the original image
    rows, cols = min(
        ((r, n // r) for r in range(1, n + 1) if n % r == 0),
        key=lambda rc: abs(np.log((rc[1] / rc[0]) / aspect)),
    )
    th, tw = crops[0].shape[:2]
    gap, sep = 6, 40
    grid_h = rows * th + (rows - 1) * gap
    grid_w = cols * tw + (cols - 1) * gap
    scale = grid_h / base.shape[0]
    from PIL import Image as _Image

    base_img = np.asarray(
        _Image.fromarray(base).resize((round(base.shape[1] * scale), grid_h))
    )
    canvas = np.full((grid_h, base_img.shape[1] + sep + grid_w, 3), 22, np.uint8)
    canvas[:, : base_img.shape[1]] = base_img
    x0 = base_img.shape[1] + sep
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        y, x = r * (th + gap), x0 + c * (tw + gap)
        canvas[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return canvas


def patch_grid(shape: tuple[int, int], patch: int) -> list[list[list[int]]]:
    h, w = shape
    strips = [[[x, 0], [x, h]] for x in range(0, w + 1, patch)]
    strips += [[[0, y], [w, y]] for y in range(0, h + 1, patch)]
    return strips
