from pathlib import Path

import click

from common.cli import serve_options

DEFAULT_PROMPTS = ("building",)
MODEL_ID = "facebook/sam3"


@click.group()
def archstil():
    """Find and segment buildings in images."""


@archstil.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=Path("buildings_out"),
    show_default=True,
    help="Directory for overlays, masks and JSON.",
)
@click.option(
    "-p",
    "--prompt",
    "prompts",
    multiple=True,
    default=DEFAULT_PROMPTS,
    show_default=True,
    help="Text concept(s) to segment; repeat for several.",
)
@click.option(
    "--threshold", default=0.5, show_default=True, help="Instance score threshold."
)
@click.option(
    "--mask-threshold",
    default=0.5,
    show_default=True,
    help="Mask binarisation threshold.",
)
@click.option(
    "--dedup-iou",
    default=0.7,
    show_default=True,
    help="Box IoU above which detections from different prompts are merged.",
)
@click.option(
    "--device", default=None, help="torch device (default: cuda if available)."
)
@click.option("--model-id", default=MODEL_ID, show_default=True)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Process a random sample of this many images.",
)
@click.option(
    "--seed", default=0, show_default=True, help="Random seed for --limit sampling."
)
@click.option(
    "--watch",
    is_flag=True,
    help="Keep segmenting new images as they appear in SOURCE (e.g. while `scrape` "
    "runs), until Ctrl+C.",
)
@click.option(
    "--interval", default=10.0, show_default=True, help="Seconds between --watch scans."
)
def segment(
    source: Path,
    output: Path,
    prompts: tuple[str, ...],
    threshold: float,
    mask_threshold: float,
    dedup_iou: float,
    device: str | None,
    model_id: str,
    limit: int | None,
    seed: int,
    watch: bool,
    interval: float,
):
    """Segment buildings in SOURCE (an image file or a directory) with SAM3.

    Images already segmented in OUTPUT are skipped, so a rerun only does new ones.

    Writes per image an overlay PNG, one mask PNG per building and a JSON file
    with boxes, scores and mask areas.
    """
    # imported here so `--help` does not pay for loading torch and transformers
    from .segment import run

    run(
        source,
        output,
        prompts,
        threshold,
        mask_threshold,
        dedup_iou,
        device,
        model_id,
        limit,
        seed,
        interval if watch else None,
    )


@archstil.command()
@click.option(
    "--buildings-only/--all-disciplines",
    default=True,
    show_default=True,
    help="Count only objects of the Bouwkundig (architectural) discipline.",
)
@click.option(
    "--all-terms",
    is_flag=True,
    help="Instead: every term of the thesaurus, prehistoric cultures included, "
    "without counts.",
)
def styles(buildings_only: bool, all_terms: bool):
    """List the architectural styles of Onroerend Erfgoed, with how many objects
    have each one.

    A style's count includes its narrower styles (indented below it), since the
    inventaris filter does too.
    """
    import asyncio

    from .onroerend import building_styles, list_styles

    if all_terms:
        for term in asyncio.run(list_styles()):
            click.echo(f"{term['id']:>5}  {term['label']}")
        return

    def show(styles, depth: int = 0) -> None:
        for style in styles:
            objects = "?" if style.count is None else style.count
            click.echo(f"{style.id:>5}  {objects:>7}  {'  ' * depth}{style.label}")
            show(style.children, depth + 1)

    click.echo(f"{'id':>5}  {'objects':>7}  style")
    show(asyncio.run(building_styles(buildings_only)))


@archstil.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=Path("archstil_references"),
    show_default=True,
    help="Directory for images/, references.jsonl and the API cache.",
)
@click.option(
    "-s",
    "--style",
    "style_names",
    multiple=True,
    help="Style label, id or URI (see `styles`); repeat for several. "
    "Without it, every object.",
)
@click.option(
    "--typologie",
    default=None,
    help="Only objects of this type: an erfgoedtypes id or URI, e.g. 513 (chapels).",
)
@click.option(
    "--limit",
    default=0,
    show_default=True,
    help="Objects to save per style, 0 for all. Gives a balanced set per style.",
)
@click.option(
    "--size",
    type=click.Choice(["medium", "large", "full"]),
    default="large",
    show_default=True,
    help="Beeldbank image size: 500 px, 1024 px or the original.",
)
@click.option(
    "--require-style/--any-style",
    default=True,
    show_default=True,
    help="Skip objects whose record has no style.",
)
@click.option(
    "--buildings-only/--all-disciplines",
    default=True,
    show_default=True,
    help="Only objects of the Bouwkundig (architectural) discipline.",
)
@click.option("--concurrency", default=4, show_default=True, help="Requests in flight.")
@click.option(
    "--rate", default=2.0, show_default=True, help="Requests per second per host."
)
def scrape(
    output: Path,
    style_names: tuple[str, ...],
    typologie: str | None,
    limit: int,
    size: str,
    require_style: bool,
    buildings_only: bool,
    concurrency: int,
    rate: float,
):
    """Download reference photos and their style from Onroerend Erfgoed.

    Writes the photos to OUTPUT/images and one JSON line per heritage object, with
    its styles, dating, type, location and the photo's licence and attribution, to
    OUTPUT/references.jsonl. Rerun to resume: objects already saved are skipped and
    every API answer is cached under OUTPUT/cache.

    Mind the fair use policy: https://developers.onroerenderfgoed.be/docs/fair-use
    """
    import asyncio

    from .onroerend import scrape as run

    if typologie and not typologie.startswith("http"):
        typologie = f"https://id.erfgoed.net/thesauri/erfgoedtypes/{typologie}"
    try:
        asyncio.run(
            run(
                output,
                style_names,
                typologie,
                limit,
                size,
                concurrency,
                rate,
                require_style,
                buildings_only,
            )
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def find_references(segments: Path, references: Path | None) -> Path:
    """`references`, or the references.jsonl in SEGMENTS or else its parent, which
    `scrape` may not have written yet: the viewer reads it once it is there."""
    if references is not None:
        return references
    inside = segments / "references.jsonl"
    return inside if inside.is_file() else segments.parent / "references.jsonl"


def find_master(references: Path, master: Path | None) -> Path:
    """`master`, or master.json next to the references."""
    return master if master is not None else references.parent / "master.json"


segments_argument = click.argument(
    "segments", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
references_option = click.option(
    "-r",
    "--references",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="references.jsonl of `scrape`. Default: the one in SEGMENTS or its parent.",
)
master_option = click.option(
    "-m",
    "--master",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="The master file. Default: master.json next to the references.",
)


@archstil.command()
@click.argument("segments", type=click.Path(file_okay=False, path_type=Path))
@references_option
@master_option
@serve_options(5004)
def verify(segments: Path, references: Path | None, master: Path | None, **server):
    """Pick, per photo, the SAM3 segment that is the heritage object, and its class.

    SEGMENTS is the output directory of `segment`. It may still be filling up: run
    `scrape` and `segment --watch` alongside and new photos show up as they are done.
    Choices are saved to SEGMENTS/selections.json as you go, and the master file
    (metadata, class, box and polygon of every checked photo) is written on every
    save.
    """
    from common import run_app

    from .web import create_app

    segments.mkdir(parents=True, exist_ok=True)  # may start before `segment` does
    references = find_references(segments, references)
    run_app(
        create_app(segments, references, find_master(references, master)),
        **server,
    )


@archstil.command("master")
@segments_argument
@references_option
@master_option
def master_command(segments: Path, references: Path | None, master: Path | None):
    """Write the master file again from SEGMENTS/selections.json."""
    import json

    from . import master as master_file

    references = find_references(segments, references)
    path = find_master(references, master)
    selections_path = segments / master_file.SELECTIONS
    if not selections_path.is_file():
        raise click.ClickException(f"No {selections_path} yet: run `verify` first")
    data = master_file.write(
        segments,
        json.loads(selections_path.read_text()),
        master_file.load_references(references),
        path,
    )
    click.echo(
        f"{data['verified']} verified, {data['rejected']} rejected, "
        f"{len(data['classes'])} classes → {path}"
    )
    for name, n in data["counts"].items():
        click.echo(f"{n:>6}  {name}")
