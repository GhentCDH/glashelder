from pathlib import Path

import click

DEFAULT_PROMPTS = ("building",)
MODEL_ID = "facebook/sam3"


@click.group()
def architecture_identifier():
    """Find and segment buildings in images."""


@architecture_identifier.command()
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
):
    """Segment buildings in SOURCE (an image file or a directory) with SAM3.

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
    )
