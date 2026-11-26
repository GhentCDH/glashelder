from pathlib import Path

import click

from common import run_app
from common.cli import serve_options


@click.group()
def unstir():
    """Match glass plates to reference photos of the same building."""


def cluster_ids(ctx, param, values: tuple[str, ...]) -> frozenset[int]:
    """`--cluster 53,60 --cluster 61` as {53, 60, 61}."""
    try:
        return frozenset(
            int(part) for value in values for part in value.split(",") if part.strip()
        )
    except ValueError as error:
        raise click.BadParameter("cluster ids are numbers, such as 53,60,61") from error


def required(**values) -> None:
    missing = [name for name, value in values.items() if value is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise click.UsageError(f"give {flags}, or set it in glashelder.toml")


def band(ctx, param, value: str) -> tuple[int, int]:
    try:
        low, high = (int(part) for part in value.split(","))
    except ValueError as error:
        raise click.BadParameter("two numbers, such as 100,300") from error
    if low >= high:
        raise click.BadParameter("the first number must be the lower one")
    return low, high


@unstir.command()
@click.argument("scores", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for the new pairs.jsonl and meta.json",
)
@click.option(
    "--labels",
    "label_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Labels files whose pairs are skipped [default: every *.labels.jsonl "
    "in the directories next to the one SCORES is in]",
)
@click.option("--size", default=100, show_default=True, help="Pairs to write")
@click.option(
    "--multi",
    default=40,
    show_default=True,
    help="At most this many pairs of plates with more than one strong reference",
)
@click.option(
    "--strong",
    default=150,
    show_default=True,
    help="f_inliers of a strong reference",
)
@click.option(
    "--band",
    "band_",
    default="100,300",
    show_default=True,
    callback=band,
    help="f_inliers range the other pairs come from, low,high",
)
@click.option(
    "--step",
    default=25,
    show_default=True,
    help="The band pairs are spread evenly over f_inliers steps of this size",
)
@click.option("--seed", default=0, show_default=True)
def sample(
    scores: Path,
    output: Path,
    label_files: tuple[Path, ...],
    size: int,
    multi: int,
    strong: int,
    band_: tuple[int, int],
    step: int,
    seed: int,
):
    """Take a next round of pairs to label from an earlier `unstir pairs` run.

    SCORES is its scores.parquet, or its directory. Pairs that have a label are
    skipped. No matching runs again. Then `unstir serve OUTPUT/pairs.jsonl`.
    """
    from .rounds import Plan, run

    if scores.is_dir():
        scores = scores / "scores.parquet"
    labels = list(label_files) or sorted(
        scores.resolve().parent.parent.glob("*/*.labels.jsonl")
    )
    plan = Plan(size=size, multi=multi, strong=strong, band=band_, step=step, seed=seed)
    run(scores, labels, output, plan)


@unstir.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for scores.parquet, pairs.jsonl and meta.json",
)
@click.option(
    "--cluster",
    "clusters_ids",
    multiple=True,
    required=True,
    callback=cluster_ids,
    help="Ids of the clusters the plates come from, comma separated or repeated",
)
@click.option(
    "--clusters",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clustering result [paths.clusters]",
)
@click.option(
    "--plate-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Plate embeddings for retrieval [unstir.plate_embeddings]",
)
@click.option(
    "--plate-images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory of the plate images [paths.images.glashelder]",
)
@click.option(
    "--reference-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference embeddings, same model as the plates [unstir.reference_embeddings]",
)
@click.option(
    "--reference-images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory of the reference photos [unstir.reference_images]",
)
@click.option(
    "--top-k", default=20, show_default=True, help="Candidate references per plate"
)
@click.option(
    "--plates",
    default=0,
    show_default=True,
    help="Use at most this many plates of the clusters, at random; 0 for all",
)
@click.option(
    "--sample",
    default=180,
    show_default=True,
    help="Pairs in pairs.jsonl, spread evenly over the `cells` scores",
)
@click.option(
    "--outside",
    default=20,
    show_default=True,
    help="Extra pairs from below the top k, to see what retrieval misses",
)
@click.option("--seed", default=0, show_default=True)
@click.option("--resize", default=1024, show_default=True, help="Long side, pixels")
@click.option("--max-keypoints", default=2048, show_default=True)
@click.option(
    "--f-px",
    default=4.0,
    show_default=True,
    help="Fundamental matrix inlier threshold, pixels at --resize",
)
@click.option(
    "--h-px",
    default=8.0,
    show_default=True,
    help="Homography inlier threshold, pixels at --resize",
)
def pairs(
    output: Path,
    clusters_ids: frozenset[int],
    clusters: Path | None,
    plate_embeddings: Path | None,
    plate_images: Path | None,
    reference_embeddings: Path | None,
    reference_images: Path | None,
    top_k: int,
    plates: int,
    sample: int,
    outside: int,
    seed: int,
    resize: int,
    max_keypoints: int,
    f_px: float,
    h_px: float,
):
    """Find and score candidate pairs, and write a sample of them to label.

    For each plate in the given clusters: its --top-k nearest references by
    embedding, matched with ALIKED and LightGlue and checked with a fundamental
    matrix. Then `unstir serve OUTPUT/pairs.jsonl` to label the sample.
    """
    from .match import Settings
    from .pairs import Options, run

    required(
        clusters=clusters,
        plate_embeddings=plate_embeddings,
        plate_images=plate_images,
        reference_embeddings=reference_embeddings,
        reference_images=reference_images,
    )
    assert clusters and plate_embeddings and plate_images
    assert reference_embeddings and reference_images
    options = Options(
        clusters=Path(clusters),
        cluster_ids=clusters_ids,
        plate_embeddings=Path(plate_embeddings),
        plate_images=Path(plate_images),
        reference_embeddings=Path(reference_embeddings),
        reference_images=Path(reference_images),
        top_k=top_k,
        plates=plates,
        sample=sample,
        outside=outside,
        seed=seed,
        settings=Settings(
            resize=resize, max_keypoints=max_keypoints, f_px=f_px, h_px=h_px
        ),
    )
    run(options, output)


@unstir.command()
@click.argument("pairs", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--labels",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSONL file the labels are added to [default: <PAIRS>.labels.jsonl]",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    show_default=True,
    help="Show the pairs in a random order that stays the same between runs, "
    "so the order of the pairs file (often by score) does not steer the label",
)
@click.option("--seed", default=0, show_default=True, help="Seed of the shuffle")
@click.option(
    "--show-info",
    is_flag=True,
    help="Also show the other fields of each pair, such as scores",
)
@serve_options(5004)
def serve(
    pairs: Path,
    labels: Path | None,
    shuffle: bool,
    seed: int,
    show_info: bool,
    **server,
):
    """Label the pairs in PAIRS as match or no match, in the browser.

    PAIRS is JSONL, one `{"plate": ..., "reference": ..., ...}` per line, with image
    paths (relative to PAIRS) or URLs. Same building, from any viewpoint, is a match.

    \b
    keys: m match, n no match, x clear, left and right to move,
          space for the next pair without a label
    """
    from .web import create_app

    if labels is None:
        labels = pairs.with_name(f"{pairs.stem}.labels.jsonl")
    app = create_app(pairs, labels, shuffle=shuffle, seed=seed, show_info=show_info)
    run_app(app, **server)
