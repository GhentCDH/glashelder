from pathlib import Path

import click

from common import run_app
from common.cli import options, serve_options


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


def names(ctx, param, values: tuple[str, ...]) -> frozenset[str]:
    """`--group a,b --group c` as {a, b, c}."""
    return frozenset(
        part.strip() for value in values for part in value.split(",") if part.strip()
    )


def band(ctx, param, value: str) -> tuple[int, int]:
    try:
        low, high = (int(part) for part in value.split(","))
    except ValueError as error:
        raise click.BadParameter("two numbers, such as 100,300") from error
    if low >= high:
        raise click.BadParameter("the first number must be the lower one")
    return low, high


def required(**values) -> None:
    missing = [name for name, value in values.items() if value is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise click.UsageError(f"give {flags}, or set it in glashelder.toml")


store_option = click.option(
    "--store",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Match store, a .matches directory [unstir.store]",
)

image_options = options(
    click.option(
        "--plate-images",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Directory of the plate images [paths.images.glashelder]",
    ),
    click.option(
        "--reference-images",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Directory of the reference photos; its provenance.jsonl adds "
        "names to the decisions [unstir.reference_images]",
    ),
)

rule_options = options(
    click.option("--min-f", default=150, show_default=True, help="Least f_inliers"),
    click.option(
        "--min-ratio",
        default=0.8,
        show_default=True,
        help="Least share of the matches that are F inliers",
    ),
    click.option(
        "--margin",
        default=50,
        show_default=True,
        help="Least f_inliers more than the next best reference of the plate",
    ),
)


@unstir.command()
@store_option
@click.option(
    "--cluster",
    multiple=True,
    callback=cluster_ids,
    help="Ids of the clusters the plates come from, comma separated or repeated; "
    "none for every plate [unstir.cluster_ids]",
)
@click.option(
    "--group",
    multiple=True,
    callback=names,
    help="Cluster groups from the clustering's annotations (`cluster serve`) "
    "whose plates are used, comma separated or repeated [unstir.groups]",
)
@click.option(
    "--without-group",
    multiple=True,
    callback=names,
    help="Groups whose clusters are left out, also when --cluster or --group "
    "names them [unstir.without_groups]",
)
@click.option(
    "--soft/--no-soft",
    default=False,
    show_default=True,
    help="Also use the noise plates whose strongest soft cluster is chosen "
    "[unstir.soft]",
)
@click.option(
    "--clusters",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Clustering result the clusters and groups belong to [paths.clusters]",
)
@click.option(
    "--plate-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Plate embeddings for retrieval [unstir.plate_embeddings]",
)
@click.option(
    "--reference-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference embeddings, same model as the plates [unstir.reference_embeddings]",
)
@image_options
@click.option(
    "--top-k", default=20, show_default=True, help="Candidate references per plate"
)
@rule_options
@click.option(
    "--plates",
    default=0,
    show_default=True,
    help="Use at most this many of the plates, at random; 0 for all",
)
@click.option(
    "--batch",
    default=2000,
    show_default=True,
    help="References per commit to the store; a stopped run loses at most one batch",
)
@click.option(
    "--chunk",
    default=10_000,
    show_default=True,
    help="Plates kept on the GPU at once, about 0.5 MB each; more plates than "
    "this read the references again for every chunk",
)
@click.option(
    "--rescore",
    is_flag=True,
    help="Throw the stored scores away when the matching settings changed",
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
def match(
    store: Path | None,
    cluster: frozenset[int],
    group: frozenset[str],
    without_group: frozenset[str],
    soft: bool,
    clusters: Path | None,
    plate_embeddings: Path | None,
    reference_embeddings: Path | None,
    plate_images: Path | None,
    reference_images: Path | None,
    top_k: int,
    min_f: int,
    min_ratio: float,
    margin: int,
    plates: int,
    batch: int,
    chunk: int,
    rescore: bool,
    seed: int,
    resize: int,
    max_keypoints: int,
    f_px: float,
    h_px: float,
):
    """Find the reference photo each plate shows, if any.

    \b
    1. retrieval: the --top-k nearest references of each plate, by embedding
    2. matching: ALIKED + LightGlue and a fundamental matrix, only for the
       pairs the store has no score for yet
    3. decision: the best reference of a plate is a match when it passes
       --min-f and --min-ratio and beats the next one by --margin

    The store keeps every score, so a run after references were added only
    matches the new pairs. The result is STORE/decisions.csv; check the matches
    with `unstir serve STORE/matches.jsonl`.
    """
    from .decide import Rule
    from .match import Settings
    from .pipeline import Options, Selection, run

    required(
        store=store,
        plate_embeddings=plate_embeddings,
        plate_images=plate_images,
        reference_embeddings=reference_embeddings,
        reference_images=reference_images,
    )
    assert store and plate_embeddings and plate_images
    assert reference_embeddings and reference_images
    run(
        Options(
            store=Path(store),
            selection=Selection(
                clusters=Path(clusters) if clusters else None,
                ids=cluster,
                groups=group,
                without=without_group,
                soft=soft,
            ),
            plate_embeddings=Path(plate_embeddings),
            plate_images=Path(plate_images),
            reference_embeddings=Path(reference_embeddings),
            reference_images=Path(reference_images),
            top_k=top_k,
            plates=plates,
            batch=batch,
            chunk=chunk,
            seed=seed,
            rescore=rescore,
            settings=Settings(
                resize=resize, max_keypoints=max_keypoints, f_px=f_px, h_px=h_px
            ),
            rule=Rule(min_f=min_f, min_ratio=min_ratio, margin=margin),
        )
    )


@unstir.command()
@store_option
@image_options
@rule_options
def decide(
    store: Path | None,
    plate_images: Path | None,
    reference_images: Path | None,
    min_f: int,
    min_ratio: float,
    margin: int,
):
    """Decide again from the stored scores, for example with other thresholds.

    No matching runs. Writes STORE/decisions.csv and STORE/matches.jsonl.
    """
    from . import decide as deciding
    from .store import MatchStore

    required(store=store, plate_images=plate_images, reference_images=reference_images)
    assert store and plate_images and reference_images
    deciding.run(
        MatchStore.open(store),
        deciding.Rule(min_f=min_f, min_ratio=min_ratio, margin=margin),
        Path(plate_images),
        Path(reference_images),
    )


@unstir.command()
@store_option
@image_options
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
    "in the directories next to OUTPUT]",
)
@click.option("--size", default=100, show_default=True, help="Pairs to write")
@click.option(
    "--multi",
    default=40,
    show_default=True,
    help="At most this many pairs of plates with more than one strong reference",
)
@click.option(
    "--strong", default=150, show_default=True, help="f_inliers of a strong reference"
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
    store: Path | None,
    plate_images: Path | None,
    reference_images: Path | None,
    output: Path,
    label_files: tuple[Path, ...],
    size: int,
    multi: int,
    strong: int,
    band_: tuple[int, int],
    step: int,
    seed: int,
):
    """Take a round of pairs to label from the candidates in the store.

    Pairs that have a label are skipped. No matching runs. Then
    `unstir serve OUTPUT/pairs.jsonl`.
    """
    from .rounds import Plan, run
    from .store import MatchStore

    required(store=store, plate_images=plate_images, reference_images=reference_images)
    assert store and plate_images and reference_images
    labels = list(label_files) or sorted(
        output.resolve().parent.glob("*/*.labels.jsonl")
    )
    rows = MatchStore.open(store).rows(Path(plate_images), Path(reference_images))
    plan = Plan(size=size, multi=multi, strong=strong, band=band_, step=step, seed=seed)
    run(rows, labels, output, plan)


@unstir.command()
@click.option(
    "--plate-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Plate embeddings [unstir.plate_embeddings]",
)
@click.option(
    "--reference-embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference embeddings, same model as the plates [unstir.reference_embeddings]",
)
@image_options
@click.option(
    "--top-k",
    default=10,
    show_default=True,
    help="Nearest images per collection that are checked point by point",
)
@click.option(
    "--show", default=5, show_default=True, help="Images shown per collection"
)
@rule_options
@serve_options(5005)
def lookup(
    plate_embeddings: Path | None,
    reference_embeddings: Path | None,
    plate_images: Path | None,
    reference_images: Path | None,
    top_k: int,
    show: int,
    min_f: int,
    min_ratio: float,
    margin: int,
    **server,
):
    """A web page that finds the building in an uploaded photo.

    The photo is matched like `unstir match` against the reference photos and
    the glass plates, and the page says per collection if there is a match.
    """
    from .decide import Rule
    from .lookup import Lookup, Source, create_app
    from .match import Settings

    required(
        plate_embeddings=plate_embeddings,
        plate_images=plate_images,
        reference_embeddings=reference_embeddings,
        reference_images=reference_images,
    )
    assert plate_embeddings and plate_images
    assert reference_embeddings and reference_images
    sources = [
        Source(
            "references",
            "Onroerend Erfgoed",
            Path(reference_embeddings),
            Path(reference_images),
        ),
        Source(
            "plates", "the glass plates", Path(plate_embeddings), Path(plate_images)
        ),
    ]
    found = Lookup(
        sources,
        top_k=top_k,
        show=show,
        rule=Rule(min_f=min_f, min_ratio=min_ratio, margin=margin),
        settings=Settings(),
    )
    run_app(create_app(found), **server)


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
