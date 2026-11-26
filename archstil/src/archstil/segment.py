"""Segment buildings in images with SAM3 (text-prompted concept segmentation)."""

import json
import random
import shutil
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from rich.console import Console
from rich.progress import track
from transformers import Sam3Model, Sam3Processor

from .cli import DEFAULT_PROMPTS, MODEL_ID

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path) -> Image.Image:
    # Glass plate scans are often greyscale / 16-bit: normalise to 8-bit RGB.
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(image, dtype=np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0) * 255
        image = Image.fromarray(arr.astype(np.uint8))
    return image.convert("RGB")


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)


def deduplicate(instances: list[dict], iou_threshold: float) -> list[dict]:
    """Drop overlapping detections from different prompts, keeping the best score."""
    if len(instances) < 2:
        return instances
    instances = sorted(instances, key=lambda d: d["score"], reverse=True)
    boxes = torch.tensor([d["box"] for d in instances], dtype=torch.float32)
    iou = box_iou(boxes, boxes)
    keep: list[int] = []
    for i in range(len(instances)):
        if all(iou[i, j] < iou_threshold for j in keep):
            keep.append(i)
    return [instances[i] for i in keep]


class BuildingSegmenter:
    def __init__(self, device: str, dtype: torch.dtype, model_id: str = MODEL_ID):
        self.device = device
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        prompts: tuple[str, ...],
        threshold: float,
        mask_threshold: float,
    ) -> list[dict]:
        instances: list[dict] = []
        for prompt in prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            for mask, box, score in zip(
                result["masks"], result["boxes"], result["scores"], strict=True
            ):
                instances.append(
                    {
                        "prompt": prompt,
                        "score": float(score),
                        "box": [float(v) for v in box.tolist()],
                        "mask": mask.bool().cpu().numpy(),
                    }
                )
        return instances


def render_overlay(image: Image.Image, instances: list[dict]) -> Image.Image:
    rng = np.random.default_rng(0)
    overlay = np.asarray(image, dtype=np.float32).copy()
    colours = [tuple(int(c) for c in rng.integers(40, 255, 3)) for _ in instances]
    for inst, colour in zip(instances, colours, strict=True):
        overlay[inst["mask"]] = overlay[inst["mask"]] * 0.5 + np.array(colour) * 0.5
    out = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    width = max(2, image.width // 400)
    for inst, colour in zip(instances, colours, strict=True):
        draw.rectangle(inst["box"], outline=colour, width=width)
        x0, y0 = inst["box"][:2]
        draw.text(
            (x0 + 4, y0 + 4), f"{inst['prompt']} {inst['score']:.2f}", fill=colour
        )
    return out


def save_results(
    path: Path, image: Image.Image, instances: list[dict], output: Path
) -> None:
    stem = path.stem
    original = output / path.name
    if original.resolve() != path.resolve():
        shutil.copy2(path, original)
    render_overlay(image, instances).save(output / f"{stem}_overlay.png")

    mask_dir = output / f"{stem}_masks"
    if instances:
        mask_dir.mkdir(exist_ok=True)
    records = []
    for i, inst in enumerate(instances):
        Image.fromarray(inst["mask"].astype(np.uint8) * 255).save(mask_dir / f"{i}.png")
        records.append(
            {
                "id": i,
                "prompt": inst["prompt"],
                "score": round(inst["score"], 4),
                "box_xyxy": [round(v, 1) for v in inst["box"]],
                "mask_area_px": int(inst["mask"].sum()),
                "mask_area_frac": round(float(inst["mask"].mean()), 5),
                "mask": f"{mask_dir.name}/{i}.png",
            }
        )
    (output / f"{stem}.json").write_text(
        json.dumps(
            {
                "image": str(path),
                "original": original.name,
                "width": image.width,
                "height": image.height,
                "buildings": records,
            },
            indent=2,
        )
    )


def run(
    source: Path,
    output: Path,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    dedup_iou: float = 0.7,
    device: str | None = None,
    model_id: str = MODEL_ID,
    limit: int | None = None,
    seed: int = 0,
) -> None:
    """Segment buildings in `source` (an image file or a directory of images)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    images = find_images(source)
    if not images:
        raise click.ClickException(f"No images found in {source}")
    if limit is not None and limit < len(images):
        images = sorted(random.Random(seed).sample(images, limit))
    output.mkdir(parents=True, exist_ok=True)

    console.print(f"Loading [bold]{model_id}[/] on {device} ({dtype})")
    segmenter = BuildingSegmenter(device, dtype, model_id)

    total = 0
    for path in track(images, description="Segmenting", console=console):
        try:
            image = load_image(path)
        except OSError as e:
            console.print(f"[yellow]Skipping {path}: {e}")
            continue
        instances = segmenter.segment(image, prompts, threshold, mask_threshold)
        if len(prompts) > 1:
            instances = deduplicate(instances, dedup_iou)
        save_results(path, image, instances, output)
        total += len(instances)

    console.print(f"Found {total} building(s) in {len(images)} image(s) → {output}")
