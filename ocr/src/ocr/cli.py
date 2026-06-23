import base64
import io
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.json import JSON
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ocr.annotation import Annotation, annotation_from_ocr, parse_buffer, render_buffer
from preprocessing.bounding import BoundingBox, load_bounding_boxes

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger("ocr")

# fraction of a dimension a box may miss and still count as "the whole image"
FULL_IMAGE_TOL = 0.02

# enforced output shape; vLLM constrains decoding to match (guided/structured output)
_TEXT_ENTRY = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
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


def _list_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        data = json.load(resp)
    return [m["id"] for m in data["data"]]


def _pick_model(base_url: str, console: Console) -> str:
    from rich.prompt import IntPrompt

    models = _list_models(base_url)
    if not models:
        log.error(f"no models running at {base_url}")
        raise SystemExit(1)
    if len(models) == 1:
        log.info(f"using only available model {models[0]!r}")
        return models[0]

    log.info(f"{len(models)} models available at {base_url}")
    for i, m in enumerate(models):
        console.print(f"  [cyan]{i}[/cyan] {m}")
    idx = IntPrompt.ask("select model", choices=[str(i) for i in range(len(models))])
    return models[idx]


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


def _build_content(image: Path, box: BoundingBox | None) -> list[dict]:
    """Build the chat 'content' parts (prompt + image(s)) for one image."""
    from PIL import Image, ImageFilter

    from ocr.prompt import CONTENT_ONLY_PROMPT, PROMPT, TWO_IMAGE_PROMPT

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
    region = (box.x, box.y, box.x + box.w, box.y + box.h)
    crop = img.crop(region)

    blurred = img.copy()
    blurred.paste(crop.filter(ImageFilter.GaussianBlur(20)), region)

    return [
        {"type": "text", "text": TWO_IMAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(blurred)}},
        {"type": "image_url", "image_url": {"url": _data_url(crop)}},
    ]


def _ocr_image(
    image: Path,
    base_url: str,
    model: str,
    box: BoundingBox | None = None,
    include_input: bool = False,
) -> str:
    """OCR a single image via the vLLM chat-completions endpoint; returns JSON text.

    With include_input, wraps as {"input": <chat content>, "output": <model json>};
    otherwise returns the raw model output text.
    """
    log.info(f"reading image {str(image)!r}")
    content = _build_content(image, box)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ocr_output", "schema": OUTPUT_SCHEMA},
        },
        "temperature": 0,
    }

    log.info(f"sending request to {base_url}/v1/chat/completions")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
    text = result["choices"][0]["message"]["content"]

    if not include_input:
        return text
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = text  # keep raw if model broke the schema
    return json.dumps({"input": content, "output": out}, ensure_ascii=False)


def _print_result(console: Console, text: str):
    try:
        console.print(JSON(text))
    except ValueError:
        log.warning("model output was not valid JSON, printing raw")
        console.print(text)


@click.group()
def ocr():
    pass


@ocr.command()
@click.option(
    "-i",
    "--image",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path for input image to do OCR on",
)
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="vLLM service base URL",
)
@click.option(
    "-b",
    "--boxes",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="BoundingBox JSON; box (keyed by image stem) drives blur/crop strategy",
)
@click.option("--model", default=None, help="Model id (skips interactive selection)")
@click.option(
    "--include-input",
    is_flag=True,
    help='Wrap output as {"input": <chat content>, "output": <model json>}',
)
def one(
    image: Path,
    base_url: str,
    model: str | None,
    boxes: Path | None,
    include_input: bool,
):
    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box = load_bounding_boxes(boxes).get(image.stem) if boxes else None
    _print_result(console, _ocr_image(image, base_url, model, box, include_input))


@ocr.command()
@click.option(
    "-i",
    "--images",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of input images to do OCR on",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write output jsons",
)
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="vLLM service base URL",
)
@click.option(
    "-b",
    "--boxes",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="BoundingBox JSON; box (keyed by image stem) drives blur/crop strategy",
)
@click.option("--model", default=None, help="Model id (skips interactive selection)")
@click.option(
    "--include-input",
    is_flag=True,
    help='Wrap output as {"input": <chat content>, "output": <model json>}',
)
def many(
    images: Path,
    output: Path,
    base_url: str,
    model: str | None,
    boxes: Path | None,
    include_input: bool,
):
    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box_map = load_bounding_boxes(boxes) if boxes else {}
    output.mkdir(exist_ok=True)

    paths = sorted(images.glob("*.jpg"))
    log.info(f"{len(paths)} images found in {str(images)!r}")
    for image in paths:
        text = _ocr_image(
            image, base_url, model, box_map.get(image.stem), include_input
        )
        _print_result(console, text)

        out = output / f"{image.stem}.json"
        out.write_text(text)
        log.info(f"wrote {str(out)!r}")


# how many candidates to buffer before shuffling; gives random order without
# walking the whole (network) share up front
_SHUFFLE_BUFFER = 512


def _candidate_images(images: Path, output: Path, buffer: int = _SHUFFLE_BUFFER):
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


@ocr.command()
@click.option(
    "-i",
    "--images",
    default=lambda: os.environ.get("ASSETS_DIR"),
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images to annotate [default: $ASSETS_DIR]",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write annotation jsons (same stem as image)",
)
@click.option(
    "-n", "--count", default=50, show_default=True, help="Max images this session"
)
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="vLLM service base URL (used to pre-fill the editor)",
)
@click.option(
    "-b",
    "--boxes",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="BoundingBox JSON; box (keyed by image stem) drives blur/crop strategy",
)
@click.option("--model", default=None, help="Model id (skips interactive selection)")
def annotate(
    images: Path | None,
    output: Path,
    count: int,
    base_url: str,
    boxes: Path | None,
    model: str | None,
):
    """Hand-transcribe images in an editor, pre-filled by the VLM, one at a time."""
    if images is None:
        raise click.UsageError("no --images given and $ASSETS_DIR is not set")

    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box_map = load_bounding_boxes(boxes) if boxes else {}
    output.mkdir(exist_ok=True)

    # Copy each image off the (autofs) share to local temp before the viewer/OCR touch
    # it. Loupe and PIL hold files open; keeping those handles on the mount stops autofs
    # from idle-unmounting. Working on a local copy means only the brief copy reads the
    # mount. The dir is wiped on exit.
    tmpdir = Path(tempfile.mkdtemp(prefix="ocr-annotate-"))

    def seed(image: Path) -> tuple[Path | None, Annotation]:
        """Copy image local, then VLM-transcribe it. Returns (local_path, pre-fill)."""
        try:
            local = tmpdir / image.name
            shutil.copy(image, local)
        except Exception as e:
            log.warning(f"copy failed for {image.name}: {e}")
            return None, Annotation()
        try:
            text = _ocr_image(local, base_url, model, box_map.get(image.stem))
            return local, annotation_from_ocr(text)
        except Exception as e:
            log.warning(f"VLM seed failed for {image.stem}: {e}")
            return local, Annotation()

    candidates = _candidate_images(images, output)
    cur = next(candidates, None)
    if cur is None:
        log.info("no un-annotated images found, nothing to do")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(seed, cur)  # kick off image 1
            while cur is not None and done < count:
                nxt = next(candidates, None)
                nxt_fut = (
                    ex.submit(seed, nxt) if nxt else None
                )  # start N+1 before editing N

                local, ann_seed = fut.result()  # type:ignore
                view = local or cur  # fall back to mount path only if the copy failed
                log.info(f"annotating {cur.name} ({done + 1}/{count})")
                # detached: xdg-open/loupe must not block the editor (run() would wait)
                subprocess.Popen(
                    ["xdg-open", str(view)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # nvim owns the terminal; mute the background prefetch thread's logs so
                # they don't corrupt its TUI. HTTP prefetch keeps running, just silently.
                logging.disable(logging.CRITICAL)
                try:
                    edited = click.edit(
                        render_buffer(cur.name, ann_seed, done + 1, count),
                        extension=".txt",
                    )
                finally:
                    logging.disable(logging.NOTSET)

                if edited is None:  # editor aborted (:cq / unchanged) -> quit
                    log.info("editor aborted, stopping")
                    break
                ann, quit = parse_buffer(edited)
                if quit:
                    log.info("'quit' received, stopping")
                    break
                if ann.is_empty():
                    log.info(f"empty annotation for {cur.name}, skipping")
                else:
                    ann.save(output / f"{cur.stem}.json")
                    done += 1
                    log.info(f"wrote {cur.stem}.json")

                cur, fut = nxt, nxt_fut
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    log.info(f"done, {done} annotation(s) written to {str(output)!r}")


@ocr.command()
@click.option(
    "-i",
    "--images",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the images reside",
)
@click.option(
    "-l",
    "--labels",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the ground truth labels reside",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="File path to write validation report to",
)
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="vLLM service base URL (used to pre-fill the editor)",
)
@click.option(
    "-b",
    "--boxes",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="BoundingBox JSON; box (keyed by image stem) drives blur/crop strategy",
)
@click.option("--model", default=None, help="Model id (skips interactive selection)")
def validate(
    images: Path,
    labels: Path,
    output: Path,
    base_url: str,
    boxes: Path | None,
    model: str | None,
):
    from ocr.validation import validate as vd

    base_url = base_url.rstrip("/")
    console = Console()

    rich_handler = next(
        (h for h in logging.getLogger().handlers if isinstance(h, RichHandler)), None
    )
    if rich_handler:
        rich_handler.console = console

    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    pairs = {}

    gts = list(labels.glob("*.json"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting...", total=len(gts))
        for annotation in gts:
            image = images / annotation.with_suffix(".jpg").name

            if not image.exists():
                log.error(f"{image!r} doesn't exist!")
                continue

            try:
                box = load_bounding_boxes(boxes).get(image.stem) if boxes else None
                ground_truth: Annotation = Annotation.load(annotation)
                result = json.loads(_ocr_image(image, base_url, model, box, True))
                data = result["output"]
                data = {
                    "meta": [i["text"] for i in data["meta"]],
                    "content": [i["text"] for i in data["content"]],
                }
                prediction: Annotation = Annotation.from_dict(data)

                pairs[annotation.stem] = (ground_truth, prediction)
            except Exception as e:
                log.error(e)

            progress.advance(task)

    report = vd(pairs)

    with open(output, "w") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
