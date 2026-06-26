import base64
import hashlib
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
from ocr.prompt import TYPES
from preprocessing.bounding import BoundingBox, load_bounding_boxes

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


def _cache_dir() -> Path:
    """Transcription cache dir: $XDG_CACHE_HOME/glashelder or ~/.cache/glashelder."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = Path(base) / "glashelder" / "transcriptions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(payload: dict) -> str:
    """Stable hash of the full request payload (model, prompt, images, params)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _resolve_concurrency(value: int | None) -> int:
    """Parallel in-flight requests: CLI value, else $OCR_CONCURRENCY, else 1."""
    if value is None:
        value = int(os.environ.get("OCR_CONCURRENCY", "1"))
    return max(1, value)


_CONCURRENCY_OPT = click.option(
    "-j",
    "--concurrency",
    default=None,
    type=int,
    help="Parallel requests to vLLM [env: OCR_CONCURRENCY, default 1]",
)


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

    from ocr.prompt import (
        CONTENT_ONLY_PROMPT,
        PROMPT,
        TWO_IMAGE_PROMPT,
    )

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


def _build_split_contents(
    image: Path, box: BoundingBox
) -> tuple[list[dict], list[dict]]:
    """Two content lists for split mode: (meta_query, content_query)."""
    from PIL import Image, ImageFilter

    from ocr.prompt import CROP_CONTENT_PROMPT, META_ONLY_PROMPT

    img = Image.open(image)
    region = (box.x, box.y, box.x + box.w, box.y + box.h)
    crop = img.crop(region)
    blurred = img.copy()
    blurred.paste(crop.filter(ImageFilter.GaussianBlur(20)), region)

    meta_content = [
        {"type": "text", "text": META_ONLY_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(blurred)}},
    ]
    crop_content = [
        {"type": "text", "text": CROP_CONTENT_PROMPT},
        {"type": "image_url", "image_url": {"url": _data_url(crop)}},
    ]
    return meta_content, crop_content


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web; returns titles, snippets, and URLs. Use fetch_url on a result URL to read the full page.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the full text content of a URL (e.g. from a search result).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
}

_FETCH_TRUNCATE = 4000


def _fetch_url(url: str, verbose: bool = False) -> str:
    import html.parser

    class _Stripper(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style", "nav", "header", "footer"}:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in {"script", "style", "nav", "header", "footer"}:
                self._skip = False

        def handle_data(self, data):
            if not self._skip and data.strip():
                self._parts.append(data.strip())

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[fetch failed: {e}]"

    s = _Stripper()
    s.feed(raw)
    text = " ".join(s._parts)[:_FETCH_TRUNCATE]
    if verbose:
        log.info(f"  fetched {url}: {len(text)} chars")
    return text


def _run_search(query: str, sites: list[str], verbose: bool = False) -> str:
    from ddgs import DDGS

    if sites:
        site_filter = " OR ".join(f"site:{s}" for s in sites)
        query = f"({site_filter}) {query}"
    results = DDGS().text(query, max_results=5)
    lines = []
    for i, r in enumerate(results):
        lines.append(f"[{r.get('href', '?')}]\n{r['title']}\n{r['body']}")
        if verbose:
            log.info(
                f"  result: [{r.get('href', '?')}] {r['title']}\n    {r['body'][:120]}"
            )
        if i == 0 and r.get("href"):
            full = _fetch_url(r["href"], verbose)
            lines.append(f"[full page content]\n{full}")
    return "\n\n".join(lines)


def _call_vllm_with_tools(
    base_url: str,
    model: str,
    messages: list[dict],
    sites: list[str],
    max_searches: int = 0,
    no_search: bool = False,
    verbose: bool = False,
) -> str:
    searches = 0
    while True:
        allow_tools = not no_search and (max_searches == 0 or searches < max_searches)
        payload: dict = {"model": model, "messages": messages}
        if allow_tools:
            payload["tools"] = [SEARCH_TOOL]
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.load(resp)
        choice = result["choices"][0]
        messages.append(choice["message"])
        if choice["finish_reason"] != "tool_calls":
            return choice["message"]["content"] or ""
        for tc in choice["message"]["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            if tc["function"]["name"] == "fetch_url":
                log.info(f"fetch: {args['url']!r}")
                content = _fetch_url(args["url"], verbose)
            else:
                searches += 1
                log.info(f"search #{searches}: {args['query']!r}")
                content = _run_search(args["query"], sites, verbose)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
            )


def _call_vllm(base_url: str, payload: dict, use_cache: bool, label: str) -> str:
    cached = _cache_dir() / f"{_cache_key(payload)}.txt" if use_cache else None
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


def _ocr_image(
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

        def _payload(content):
            return {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "ocr_output", "schema": OUTPUT_SCHEMA},
                },
                "temperature": 0,
            }

        meta_text = _call_vllm(
            base_url, _payload(meta_content), use_cache, f"{image.stem}:meta"
        )
        crop_text = _call_vllm(
            base_url, _payload(crop_content), use_cache, f"{image.stem}:content"
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ocr_output", "schema": OUTPUT_SCHEMA},
        },
        "temperature": 0,
    }

    text = _call_vllm(base_url, payload, use_cache, image.stem)

    if not include_input:
        return text
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = text  # keep raw if model broke the schema
    return json.dumps({"input": content, "output": out}, ensure_ascii=False)


def _print_result(text: str):
    print(text)


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
@click.option(
    "--split",
    is_flag=True,
    help="Split two-image case into two separate queries (meta + content) and merge",
)
def one(
    image: Path,
    base_url: str,
    model: str | None,
    boxes: Path | None,
    include_input: bool,
    split: bool,
):
    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box = load_bounding_boxes(boxes).get(image.stem) if boxes else None
    _print_result(_ocr_image(image, base_url, model, box, include_input, split=split))


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
@click.option(
    "--split",
    is_flag=True,
    help="Split two-image case into two separate queries (meta + content) and merge",
)
@click.option(
    "--nocache",
    is_flag=True,
    help="Disable prompt cache",
)
@_CONCURRENCY_OPT
def many(
    images: Path,
    output: Path,
    base_url: str,
    model: str | None,
    boxes: Path | None,
    include_input: bool,
    split: bool,
    nocache: bool,
    concurrency: int | None,
):
    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box_map = load_bounding_boxes(boxes) if boxes else {}
    output.mkdir(exist_ok=True)

    paths = sorted(images.glob("*.jpg"))
    workers = _resolve_concurrency(concurrency)
    log.info(f"{len(paths)} images found in {str(images)!r} (concurrency {workers})")

    def process(image: Path) -> Path:
        text = _ocr_image(
            image,
            base_url,
            model,
            box_map.get(image.stem),
            include_input,
            split=split,
            use_cache=not nocache,
        )
        out = output / f"{image.stem}.json"
        out.write_text(text)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(process, paths):
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
@click.option(
    "--split",
    is_flag=True,
    help="Split two-image case into two separate queries (meta + content) and merge",
)
@_CONCURRENCY_OPT
def validate(
    images: Path,
    labels: Path,
    output: Path,
    base_url: str,
    boxes: Path | None,
    model: str | None,
    split: bool,
    concurrency: int | None,
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

    box_map = load_bounding_boxes(boxes) if boxes else {}
    gts = list(labels.glob("*.json"))
    workers = _resolve_concurrency(concurrency)

    def predict(annotation: Path) -> tuple[str, tuple[Annotation, Annotation]] | None:
        image = images / annotation.with_suffix(".jpg").name
        if not image.exists():
            log.error(f"{image!r} doesn't exist!")
            return None
        try:
            ground_truth = Annotation.load(annotation)
            result = json.loads(
                _ocr_image(
                    image, base_url, model, box_map.get(image.stem), True, split=split
                )
            )
            data = result["output"]
            data = {
                "meta": [i["text"] for i in data["meta"]],
                "content": [i["text"] for i in data["content"]],
            }
            return annotation.stem, (ground_truth, Annotation.from_dict(data))
        except Exception as e:
            log.error(e)
            return None

    pairs = {}
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
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(predict, gts):
                if res is not None:
                    pairs[res[0]] = res[1]
                progress.advance(task)

    report = vd(pairs)
    report.config.model = model
    report.config.split = split

    with open(output, "w") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)


_DEFAULT_ENRICH_PROMPT = (
    "Given this OCR result from a historical glass plate:\n{ocr}\n\n"
    "Use search to find more information about the places, creators, and years mentioned."
)


@ocr.command()
@click.argument(
    "ocr_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="vLLM service base URL",
)
@click.option("--model", default=None, help="Model id (skips interactive selection)")
@click.option(
    "-i",
    "--images",
    default=lambda: os.environ.get("ASSETS_DIR"),
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Images directory; image matching OCR JSON stem is included in prompt [default: $ASSETS_DIR]",
)
@click.option(
    "--site",
    "sites",
    multiple=True,
    metavar="DOMAIN",
    help="Restrict search to this domain (repeatable, e.g. --site en.wikipedia.org)",
)
@click.option(
    "--prompt",
    default=_DEFAULT_ENRICH_PROMPT,
    show_default=True,
    help="Prompt template; {ocr} is replaced with the OCR JSON content",
)
@click.option(
    "--max-searches",
    default=0,
    show_default=True,
    help="Max search calls the model may make (0 = unlimited)",
)
@click.option(
    "--no-search",
    is_flag=True,
    help="Disable web search entirely; model answers from image+OCR only",
)
@click.option(
    "--verbose-search",
    is_flag=True,
    help="Log each search result URL and snippet to stderr",
)
def enrich(
    ocr_json: Path,
    base_url: str,
    model: str | None,
    images: Path | None,
    sites: tuple[str, ...],
    prompt: str,
    max_searches: int,
    no_search: bool,
    verbose_search: bool,
):
    """Enrich OCR output with web search via VLM tool calling."""
    from PIL import Image

    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    raw = json.loads(ocr_json.read_text())
    # strip base64 image data when OCR was run with --include-input
    ocr_data = raw.get("output", raw)
    ocr_text = json.dumps(ocr_data, ensure_ascii=False)

    content: list[dict] = [{"type": "text", "text": prompt.format(ocr=ocr_text)}]
    if images is not None:
        image_path = images / f"{ocr_json.stem}.jpg"
        if image_path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(Image.open(image_path))},
                }
            )
            log.info(f"included image {image_path.name}")
        else:
            log.warning(f"image not found: {image_path}")

    messages = [
        {
            "role": "system",
            "content": "Je bent een assistent die historische glasplaten uit Belgische archieven onderzoekt. Zoek altijd in het Nederlands. Geef je antwoord in het Nederlands.",
        },
        {"role": "user", "content": content},
    ]
    print(
        _call_vllm_with_tools(
            base_url,
            model,
            messages,
            list(sites),
            max_searches,
            no_search,
            verbose_search,
        )
    )


@ocr.command()
@click.option(
    "-r",
    "--report",
    "report_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Existing validation report JSON to recompute",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="File path to write the rescored report to",
)
@click.option("--tau", required=True, type=float, help="New similarity threshold")
def rescore(report_path: Path, output: Path, tau: float):
    """Recompute a report at a new --tau from stored similarities; no re-transcription."""
    from ocr.validation import rescore as rs

    report = rs(json.loads(report_path.read_text()), tau)
    with open(output, "w") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    log.info(f"wrote rescored report (tau={tau}) to {str(output)!r}")
