import base64
import hashlib
import io
import json
import logging
import os
import random
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from ocr.prompt import (
    CONTENT_ONLY_PROMPT,
    CROP_CONTENT_PROMPT,
    META_ONLY_PROMPT,
    PROMPT,
    TWO_IMAGE_PROMPT,
    TYPES,
)
from preprocessing.bounding import BoundingBox

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger("ocr")

# fraction of a dimension a box may miss and still count as "the whole image"
FULL_IMAGE_TOL = 0.02

# enforced output shape; vLLM constrains decoding to match (guided/structured output)
_TEXT_ENTRY = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "type": {"type": "string", "enum": list(TYPES)},
    },
    "required": ["text", "type"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "meta": {"type": "array", "items": _TEXT_ENTRY},
        "content": {"type": "array", "items": _TEXT_ENTRY},
    },
    "required": ["meta", "content"],
    "additionalProperties": False,
}

# how many candidates to buffer before shuffling; gives random order without
# walking the whole (network) share up front
_SHUFFLE_BUFFER = 512


def cache_dir() -> Path:
    """Transcription cache dir: $XDG_CACHE_HOME/glashelder or ~/.cache/glashelder."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = Path(base) / "glashelder" / "transcriptions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(payload: dict) -> str:
    """Stable hash of the full request payload (model, prompt, images, params)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def list_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        data = json.load(resp)
    return [m["id"] for m in data["data"]]


def _data_url(img: "Image") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _is_full_image(box: BoundingBox, w: int, h: int) -> bool:
    """True when the box (nearly) spans the whole image -> no margin/meta present."""
    return (
        box.x <= FULL_IMAGE_TOL * w
        and box.y <= FULL_IMAGE_TOL * h
        and box.x + box.w >= (1 - FULL_IMAGE_TOL) * w
        and box.y + box.h >= (1 - FULL_IMAGE_TOL) * h
    )


def _blur_and_crop(img: "Image", box: BoundingBox) -> tuple["Image", "Image"]:
    """Return (blurred-full-image, crop) for a partial box: blurred meta, sharp crop."""
    from PIL import ImageFilter

    region = (box.x, box.y, box.x + box.w, box.y + box.h)
    crop = img.crop(region)
    blurred = img.copy()
    blurred.paste(crop.filter(ImageFilter.GaussianBlur(20)), region)
    return blurred, crop


def _build_content(image: Path, box: BoundingBox | None) -> list[dict]:
    """Build the chat 'content' parts (prompt + image(s)) for one image."""
    from PIL import Image

    img = Image.open(image)
    w, h = img.size

    if box is None:
        log.info(f"  {image.stem}: no bounding box, single image (meta + content)")
        return [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]

    if _is_full_image(box, w, h):
        log.info(f"  {image.stem}: box spans full image, content-only request")
        return [
            {"type": "text", "text": CONTENT_ONLY_PROMPT},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]

    log.info(f"  {image.stem}: partial box, two-image request (blurred meta + crop)")
    blurred, crop = _blur_and_crop(img, box)
    return [
        {"type": "text", "text": TWO_IMAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(blurred)}},
        {"type": "image_url", "image_url": {"url": _data_url(crop)}},
    ]


def _build_split_contents(
    image: Path, box: BoundingBox
) -> tuple[list[dict], list[dict]]:
    """Two content lists for split mode: (meta_query, content_query)."""
    from PIL import Image

    img = Image.open(image)
    blurred, crop = _blur_and_crop(img, box)

    meta_content = [
        {"type": "text", "text": META_ONLY_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(blurred)}},
    ]
    crop_content = [
        {"type": "text", "text": CROP_CONTENT_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(crop)}},
    ]
    return meta_content, crop_content


def _build_payload(model: str, content: list[dict]) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ocr_output", "schema": OUTPUT_SCHEMA},
        },
        "temperature": 0,
    }


def _call_vllm(base_url: str, payload: dict, use_cache: bool, label: str) -> str:
    cached = cache_dir() / f"{cache_key(payload)}.txt" if use_cache else None
    if cached and cached.exists():
        log.info(f"cache hit for {label}")
        return cached.read_text()
    log.info(f"sending request to {base_url}/v1/chat/completions")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
    text = result["choices"][0]["message"]["content"]
    if cached:
        log.info("saving response to cache")
        cached.write_text(text)
    return text


def ocr_image(
    image: Path,
    base_url: str,
    model: str,
    box: BoundingBox | None = None,
    include_input: bool = False,
    use_cache: bool = True,
    split: bool = False,
) -> str:
    """OCR a single image via the vLLM chat-completions endpoint; returns JSON text.

    With include_input, wraps as {"input": <chat content>, "output": <model json>};
    otherwise returns the raw model output text. Results are cached on disk keyed
    by the full request payload (temperature=0 makes them deterministic).

    When split=True and the image has a partial bounding box, two separate queries
    are issued (one for meta, one for content) and their results are merged.
    """
    log.info(f"reading image {str(image)!r}")

    # split mode: two queries for partial-box images
    if split and box is not None:
        from PIL import Image as _Pil

        _w, _h = _Pil.open(image).size
        _is_partial = not _is_full_image(box, _w, _h)
    else:
        _is_partial = False
    if _is_partial:
        log.info(f"  {image.stem}: split mode — two separate queries")
        meta_content, crop_content = _build_split_contents(image, box)

        meta_text = _call_vllm(
            base_url,
            _build_payload(model, meta_content),
            use_cache,
            f"{image.stem}:meta",
        )
        crop_text = _call_vllm(
            base_url,
            _build_payload(model, crop_content),
            use_cache,
            f"{image.stem}:content",
        )

        meta_out = json.loads(meta_text).get("meta", [])
        crop_out = json.loads(crop_text).get("content", [])
        merged = {"meta": meta_out, "content": crop_out}
        text = json.dumps(merged, ensure_ascii=False)

        if not include_input:
            return text
        return json.dumps(
            {
                "input": {"meta_query": meta_content, "content_query": crop_content},
                "output": merged,
            },
            ensure_ascii=False,
        )

    content = _build_content(image, box)
    payload = _build_payload(model, content)

    text = _call_vllm(base_url, payload, use_cache, image.stem)

    if not include_input:
        return text
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = text  # keep raw if model broke the schema
    return json.dumps({"input": content, "output": out}, ensure_ascii=False)


def candidate_images(images: Path, output: Path, buffer: int = _SHUFFLE_BUFFER):
    """Yield un-annotated jpgs in (locally) random order, lazily from the share."""
    buf = []
    for image in images.glob("*.jpg"):
        if (output / f"{image.stem}.json").exists():
            continue
        buf.append(image)
        if len(buf) >= buffer:
            random.shuffle(buf)
            yield from buf
            buf = []
    random.shuffle(buf)
    yield from buf
