import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from common.progress import progress as progress_bars
from ocr.annotation import Annotation, annotation_from_ocr, parse_buffer, render_buffer
from ocr.vllm import candidate_images, list_models, ocr_image
from preprocessing.bounding import load_bounding_boxes

log = logging.getLogger("ocr")


_CONCURRENCY_OPT = click.option(
    "-j",
    "--concurrency",
    default=1,
    show_default=True,
    type=int,
    help="Parallel requests to vLLM [ocr.concurrency]",
)


def _pick_model(base_url: str, console: Console) -> str:
    from rich.prompt import IntPrompt

    models = list_models(base_url)
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
    _print_result(ocr_image(image, base_url, model, box, include_input, split=split))


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
    concurrency: int,
):
    base_url = base_url.rstrip("/")
    console = Console()
    model = model or _pick_model(base_url, console)
    log.info(f"selected model {model!r}")

    box_map = load_bounding_boxes(boxes) if boxes else {}
    output.mkdir(exist_ok=True)

    paths = sorted(images.glob("*.jpg"))
    workers = max(1, concurrency)
    log.info(f"{len(paths)} images found in {str(images)!r} (concurrency {workers})")

    def process(image: Path) -> Path:
        text = ocr_image(
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


@ocr.command()
@click.option(
    "-i",
    "--images",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images to annotate [paths.images.glashelder]",
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
        raise click.UsageError(
            "no images: pass --images or set paths.images.glashelder"
        )

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
            text = ocr_image(local, base_url, model, box_map.get(image.stem))
            return local, annotation_from_ocr(text)
        except Exception as e:
            log.warning(f"VLM seed failed for {image.stem}: {e}")
            return local, Annotation()

    candidates = candidate_images(images, output)
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
    help="vLLM service base URL (used to produce the predictions being scored)",
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
    concurrency: int,
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
    workers = max(1, concurrency)

    def predict(annotation: Path) -> tuple[str, tuple[Annotation, Annotation]] | None:
        image = images / annotation.with_suffix(".jpg").name
        if not image.exists():
            log.error(f"{image!r} doesn't exist!")
            return None
        try:
            ground_truth = Annotation.load(annotation)
            result = json.loads(
                ocr_image(
                    image, base_url, model, box_map.get(image.stem), True, split=split
                )
            )
            prediction = annotation_from_ocr(json.dumps(result["output"]))
            return annotation.stem, (ground_truth, prediction)
        except Exception as e:
            log.error(e)
            return None

    pairs = {}
    with progress_bars(console=console) as progress:
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
