import logging
import pickle
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import click

from common import ASSETS_DIR, PROJECT_ROOT, require_auth, run_app
from common.images import is_url
from common.progress import format_bytes

from .embed import EMBEDDING_MODELS, get_device

PACKAGE_ROOT = PROJECT_ROOT / "embed"
EMBEDDINGS_PATH = PACKAGE_ROOT / "embeddings.pkl"

logger = logging.getLogger(__name__)


def display_name(ref: str) -> str:
    """The last part of an image reference, a path or a URL, for display."""
    if not is_url(ref):
        return Path(ref).name
    parts = urlsplit(ref)
    return unquote(parts.path.rstrip("/").rpartition("/")[2]) or parts.netloc


TOP_K_ERROR = "The number of results must be a positive whole number"


def parse_top_k(value: str | None, default: int = 12) -> int | None:
    """`top_k` from a form or query string, or `None` when it is not a positive
    whole number."""
    if value is None or not value.strip():
        return default
    try:
        top_k = int(value)
    except ValueError:
        return None
    return top_k if top_k > 0 else None


@click.group()
def embed():
    """Generate and manage image embeddings using various models."""
    pass


def store_options(command):
    """The options `embed artworks` and `embed images` share."""
    options = [
        click.option(
            "--output",
            type=click.Path(path_type=Path),
            default=None,
            help="Embedding store to create or extend",
        ),
        click.option(
            "--model",
            default="clip",
            show_default=True,
            type=click.Choice(sorted(EMBEDDING_MODELS)),
            help="Embedding model, must match the one an existing store was made with",
        ),
        click.option("--batch-size", default=64, show_default=True),
        click.option(
            "--limit", default=0, help="Only look at the first N images, 0 for all"
        ),
        click.option(
            "--verify",
            is_flag=True,
            help="Also read images that are already stored, embedding again the ones "
            "whose bytes changed. Rows without a hash only get their hash recorded.",
        ),
        click.option(
            "--prune",
            is_flag=True,
            help="Drop stored rows whose image is no longer in the input. "
            "Shifts the indices of the rows after them.",
        ),
        click.option(
            "--concurrency",
            type=int,
            default=None,
            help="Fetches in flight over all hosts [default: images.concurrency]",
        ),
        click.option(
            "--per-host",
            type=int,
            default=None,
            help="Fetches in flight to one host [default: images.per_host]",
        ),
        click.option(
            "--rate",
            type=float,
            default=None,
            help="Requests per second to one host, 0 for no limit [default: images.rate]. "
            "Hosts with their own [images.hosts] table keep that.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


@embed.command()
@click.argument(
    "artworks", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@store_options
def artworks(artworks: Path, output: Path | None, **options):
    """Embed scraped artworks into a store, fetching their images as needed.

    ARTWORKS is a scrape (JSON, JSONL, optionally .zst), by default `paths.artworks`.
    The store defaults to `paths.embeddings.reference`. Only images that are not
    in it yet are fetched, so running this after every scrape embeds what is new.
    """
    import artscraper

    urls = dict.fromkeys(a.image_url for a in artscraper.read(artworks) if a.image_url)
    logger.info(f"{artworks}: {len(urls):,} distinct image URLs")
    embed_refs({url: url for url in urls}, None, output, "reference", **options)


@embed.command()
@click.argument(
    "image_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@store_options
def images(image_dir: Path, output: Path | None, **options):
    """Embed a directory of JPGs into a store.

    IMAGE_DIR defaults to `paths.images.glashelder`, the store to
    `paths.embeddings.glashelder`. Images are stored relative to IMAGE_DIR, so
    the store keeps working when the directory is mounted elsewhere.
    """
    paths = sorted(image_dir.glob("**/*.jpg"))
    logger.info(f"{image_dir}: {len(paths):,} images")
    embed_refs(
        {str(path): str(path.relative_to(image_dir)) for path in paths},
        image_dir,
        output,
        "glashelder",
        **options,
    )


def embed_refs(
    refs: dict[str, str],
    root: Path | None,
    output: Path | None,
    kind: str,
    *,
    model: str,
    batch_size: int,
    limit: int,
    verify: bool,
    prune: bool,
    concurrency: int | None,
    per_host: int | None,
    rate: float | None,
) -> None:
    """Bring the store at `output` up to date with `refs` (source -> key in the store)."""
    from common.progress import progress

    from .embed import get_embedder
    from .store import Store, is_store
    from .stream import encode_into, pending

    if output is None:
        raise click.UsageError(
            f"no store given: pass --output or set paths.embeddings.{kind}"
        )
    if output.exists() and not is_store(output):
        raise click.ClickException(
            f"{output} is not an embedding store. If it is a legacy pickle, convert "
            f"it first: glashelder embed convert {output} {output.with_suffix('.store')}"
        )

    embedder = get_embedder(model)
    try:
        store = Store.create_or_open(
            output, model=embedder.name, model_id=embedder.model_id, root=root
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if prune and (removed := store.prune(set(refs.values()))):
        logger.warning(
            f"pruned {removed:,} rows no longer in the input; "
            "the rows after them moved up"
        )

    todo, skipped = pending(store, refs, verify=verify)
    if limit:
        todo = todo[:limit]
    logger.info(
        f"{output}: {len(store):,} rows stored, {len(todo):,} images to "
        f"{'check' if verify else 'embed'}"
    )
    if skipped:
        logger.info(
            f"left out {skipped:,} images that failed before for a lasting reason "
            "(like a 404); --verify tries them again"
        )
    if not todo:
        return
    log_limits(todo, per_host, rate)

    embedder.load(get_device())
    with progress(stderr=True) as bar:
        task = bar.add_task("reading", total=len(todo))
        done = encode_into(
            store,
            embedder,
            refs,
            todo,
            batch_size=batch_size,
            report=lambda p: bar.update(
                task,
                completed=p.done,
                description=f"embedded {p.embedded:,}, failed {p.failed.total():,}"
                + (f", hash added to {p.hashed:,} old rows" if verify else ""),
            ),
            concurrency=concurrency,
            per_host=per_host,
            rate=rate,
        )

    logger.info(
        f"{output}: embedded {done.embedded:,} (each with the hash of its image), "
        f"{done.loaded - done.embedded - done.hashed:,} unchanged"
        + (f", hash added to {done.hashed:,} old rows" if verify else "")
        + f"; {len(store):,} rows now"
    )
    if done.failed:
        reasons = ", ".join(f"{n:,} {why}" for why, n in done.failed.most_common())
        logger.warning(
            f"could not read {done.failed.total():,} images: {reasons}. "
            f"{len(store.failures()):,} failed for a lasting reason; they are listed "
            f"in {output / 'failures.jsonl'} and left out of the next run "
            "(--verify tries them again). The rest are tried again next time."
        )


def log_limits(sources: list[str], per_host: int | None, rate: float | None) -> None:
    """How each host will be fetched from, and how long that takes at the least."""
    from common.config import config
    from common.images import host

    settings = config().images
    slowest = 0.0
    for name, count in Counter(host(s) for s in sources if is_url(s)).most_common():
        workers, limit = settings.limits(name, per_host, rate)
        seconds = count / limit if limit else 0.0
        slowest = max(slowest, seconds)
        logger.info(
            f"{name}: {count:,} images, {workers} at a time, "
            + (f"{limit:g}/s, at least {hours(seconds)}" if limit else "no rate limit")
        )
    if slowest:
        logger.info(f"the rate limits alone make this run take {hours(slowest)}")


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


CONVERT_CHUNK = 65_536


@embed.command()
@click.argument(
    "sources", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.argument("destination", type=click.Path(path_type=Path))
@click.option(
    "--root",
    default=None,
    help="Directory the sources' local images were embedded from, when the one "
    "they record (or, for old pickles, the directory all their paths share) is wrong",
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    show_default=True,
    help="Dtype of a new store",
)
def convert(sources: tuple[Path, ...], destination: Path, root, dtype):
    """Convert embeddings between stores and legacy pickles, or merge several.

    DESTINATION is a store directory, created or appended to, or a `.pkl` file
    for the legacy format. All SOURCES must come from the same model. Row order
    is kept, so indices into a converted file stay the same.
    """
    with progress_bars() as progress:
        summary = convert_into(sources, destination, root, dtype, progress)
    # after the bars are gone, so it is the last thing on screen
    logger.info(summary)


def progress_bars():
    from rich.progress import (
        BarColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from common.progress import progress

    # no count: some bars count bytes
    return progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        stderr=True,
    )


class CountingWriter:
    """A file that advances a progress bar by every byte written to it."""

    def __init__(self, file, advance):
        self._file, self._advance = file, advance

    def write(self, data) -> int:
        self._advance(memoryview(data).nbytes)
        return self._file.write(data)


def convert_into(sources, destination: Path, root, dtype, progress) -> str:
    """Do the conversion `convert` describes, returning a summary of the result."""
    import torch

    from .embed import read_embeddings
    from .store import Store, is_store

    def described(source: Path) -> dict:
        data = read_embeddings(source, root=root, progress=progress)
        refs = data["images"]
        logger.info(
            f"{source}: {len(refs):,} rows of {data['model_id']}, root {data['root']}, "
            f"e.g. {refs[0] if refs else '-'}"
        )
        if duplicates := len(refs) - len(set(refs)):
            logger.warning(
                f"{source}: {duplicates:,} refs appear more than once; "
                "only the first row is kept, with the last vector"
            )
        return data

    if destination.suffix == ".pkl":
        parts = [described(source) for source in sources]
        if len({part["model_id"] for part in parts}) > 1:
            raise click.ClickException("the sources come from different models")
        data = {
            **{key: parts[0][key] for key in ("model", "model_id", "embedding_dim")},
            "root": parts[0]["root"],
            "images": [ref for part in parts for ref in part["images"]],
            "embeddings": torch.cat(
                [part["embeddings"].to(torch.float32) for part in parts]
            ),
        }
        # the vectors are nearly all of the file, which makes a good total
        task = progress.add_task(
            f"writing {destination.name}", total=data["embeddings"].nbytes
        )
        with destination.open("wb") as file:
            writer = CountingWriter(file, lambda n: progress.advance(task, n))
            pickle.dump(data, writer)
        progress.update(task, completed=data["embeddings"].nbytes)
        return (
            f"{destination}: {len(data['images']):,} rows, "
            f"{format_bytes(destination.stat().st_size)} as float32"
        )

    first = described(sources[0])
    try:
        store = Store.create_or_open(
            destination,
            model=first["model"],
            model_id=first["model_id"],
            root=first["root"],
            dtype=dtype,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    for source in sources:
        data = first if source == sources[0] else described(source)
        if data["model_id"] != store.model_id:
            raise click.ClickException(
                f"{source} holds {data['model_id']} embeddings, "
                f"{destination} holds {store.model_id}"
            )
        refs, embeddings = data["images"], data["embeddings"]
        if data["root"] != store.root and not all(map(is_url, refs)):
            logger.warning(
                f"{source} is relative to {data['root']}, {destination} to {store.root}"
            )
        known = Store.open(source).hashes() if is_store(source) else {}
        hashes = [known.get(ref) for ref in refs]
        task = progress.add_task(
            f"writing {source.name} into {destination.name}", total=len(refs)
        )
        for start in range(0, len(refs), CONVERT_CHUNK):
            end = start + CONVERT_CHUNK
            store.put(refs[start:end], hashes[start:end], embeddings[start:end].numpy())
            progress.update(task, completed=min(end, len(refs)))

    return (
        f"{destination}: {len(store):,} rows, {format_bytes(store.nbytes)} "
        f"as {store.dtype.name}"
    )


@embed.command()
@click.option(
    "--queries",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Store of the images to find matches for [paths.embeddings.glashelder]",
)
@click.option(
    "--references",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Store to search them in [paths.embeddings.reference]",
)
@click.option("--top-k", default=10, show_default=True, help="Matches per image")
def neighbours(queries: Path | None, references: Path | None, top_k: int):
    """Find and save the top-k references for every image in the query store.

    The result is saved in the reference store, and later runs (and
    `artmatcher serve`) only search what was added since: new references after
    `embed artworks`, new images after `embed images`. Best run on a GPU after
    every `embed artworks`, so serving starts at once anywhere.
    """
    import torch

    from .neighbours import neighbours as find
    from .search import place
    from .store import Store, is_store

    for name, path, setting in (
        ("--queries", queries, "glashelder"),
        ("--references", references, "reference"),
    ):
        if path is None:
            raise click.UsageError(
                f"pass {name} or set paths.embeddings.{setting} in glashelder.toml"
            )
        if not is_store(path):
            raise click.ClickException(
                f"{path} is not an embedding store; convert it first with "
                "`glashelder embed convert`"
            )
    assert queries is not None and references is not None
    query_store, reference_store = Store.open(queries), Store.open(references)

    device = get_device()
    ref_vectors = place(
        torch.from_numpy(reference_store.vectors()), device, label=str(references)
    )
    try:
        find(
            query_store,
            reference_store,
            torch.from_numpy(query_store.vectors()),
            ref_vectors,
            top_k,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def saved_neighbours(store) -> str:
    """The saved top-k results in a store, and how far behind each one is."""
    from .neighbours import DIRECTORY, load

    lines = []
    for path in sorted((store.path / DIRECTORY).glob("*.npz")):
        saved = load(path)
        if saved is None:
            continue
        state = (
            "rows changed since, searched again from scratch on next use"
            if saved.edits != store.edits
            else f"{len(store) - saved.rows:,} new rows to search"
            if saved.rows < len(store)
            else "up to date with this store"
        )
        lines.append(
            f"{path.stem}: {saved.queries:,} images, top {saved.indices.shape[1]}, "
            f"{state}"
        )
    return "\n         ".join(lines) or "none (glashelder embed neighbours makes one)"


@embed.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info(path: Path):
    """Show what an embeddings store or legacy pickle holds."""
    from .embed import read_embeddings
    from .store import Store, is_store

    data = read_embeddings(path)
    refs, embeddings = data["images"], data["embeddings"]
    if is_store(path):
        store = Store.open(path)
        kind = f"store (version {store.meta['version']})"
        hashed = sum(h is not None for h in store.hashes().values())
        hashed_line = f"{hashed:,} of {len(store):,}"
        failures = Counter(store.failures().values())
        failed_line = (
            f"{failures.total():,} ("
            + ", ".join(f"{n:,} {why}" for why, n in failures.most_common())
            + ")"
            if failures
            else "0"
        )
        neighbours_line = saved_neighbours(store)
    else:
        kind, hashed_line = "legacy pickle", "none (convert it to a store first)"
        failed_line, neighbours_line = "-", "-"

    rows, dim = embeddings.shape
    size = embeddings.element_size() * embeddings.nelement()
    for name, value in (
        ("path", path),
        ("format", kind),
        ("model", f"{data['model']} ({data['model_id']})"),
        ("rows", f"{rows:,} x {dim} {str(embeddings.dtype).removeprefix('torch.')}"),
        ("size", format_bytes(size)),
        ("hashed", hashed_line),
        ("failed", failed_line),
        ("top-k", neighbours_line),
        ("root", data["root"]),
        ("example", refs[0] if refs else "-"),
    ):
        click.echo(f"{name + ':':9}{value}")

    import torch

    from .search import nbytes, placement

    device = get_device()
    _, lines = placement(
        size, nbytes(embeddings, torch.float16), embeddings.dtype, device
    )
    click.echo(f"\nfor searching on this machine ({device}):")
    for _, line in lines:
        click.echo(f"  {line}")


@embed.command()
@click.argument("prompt")
@click.option(
    "--embeddings",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Embedding store or legacy pickle [paths.embeddings.glashelder]",
)
@click.option(
    "--image-dir",
    default=ASSETS_DIR,
    help="Directory the embedded images live in on this machine",
)
@click.option("--top-k", default=10, show_default=True, help="Matches to return")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("search_results"),
    show_default=True,
    help="Directory to copy the matches into, must be empty or not exist yet",
)
def search(
    prompt: str,
    embeddings: Path | None,
    image_dir: str,
    top_k: int,
    output_dir: Path,
):
    """Find the images that match a text PROMPT best, and copy them to a directory.

    The files are named after their rank and similarity, so the directory
    sorts best first.
    """
    import shutil

    from common.images import load

    from .embed import get_embedder, load_embeddings
    from .search import perform_search, place

    if embeddings is None:
        raise click.UsageError(
            "pass --embeddings or set paths.embeddings.glashelder in glashelder.toml"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise click.ClickException(
            f"{output_dir} is not empty; pass another --output-dir or empty it first"
        )

    device = get_device()
    data = load_embeddings(embeddings, root=image_dir)
    refs = data["images"]
    image_embeddings = place(data["embeddings"], device, label=str(embeddings))

    embedder = get_embedder(data.get("model", "clip"))
    if not embedder.supports_text:
        raise click.ClickException(
            f"{embeddings} was made with '{embedder.name}', which cannot embed text"
        )
    embedder.load(device)

    indices, scores = perform_search(
        top_k, embedder.encode_text(prompt), image_embeddings, refs
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Top {top_k} matches for '{prompt}':")
    for rank, (index, score) in enumerate(
        zip(indices.tolist(), scores.tolist(), strict=True), 1
    ):
        ref = refs[index]
        prefix = f"{rank:02d}_score{score:.3f}"
        name = display_name(ref)
        if is_url(ref):
            load(ref).save(output_dir / f"{prefix}_{Path(name).stem}.jpg")
        else:
            shutil.copy(ref, output_dir / f"{prefix}_{name}")
        logger.info(f"{rank}. {ref} (similarity: {score:.4f})")
    logger.info(f"Saved the matches to {output_dir}")


@embed.command()
@click.option(
    "--embeddings",
    default=EMBEDDINGS_PATH,
    help="Embedding store or legacy pickle [paths.embeddings.glashelder]",
)
@click.option(
    "--image-dir",
    default=ASSETS_DIR,
    help="Directory the embedded images live in on this machine",
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the site in a browser on startup",
)
def serve(embeddings, image_dir, host, port, open_browser):
    """Start a web server for interactive image search.

    Authentication is required. Set the `secret` config value (or
    GLASHELDER_SECRET), then access the server with /?secret=<your-secret> to
    get a session cookie.
    """
    import io

    from flask import Flask, redirect, render_template, request, send_from_directory
    from PIL import Image

    from .embed import get_embedder, load_embeddings
    from .search import find_similar, perform_search, place

    app = Flask(
        __name__,
        template_folder=PACKAGE_ROOT / "templates",
        static_folder=PACKAGE_ROOT / "static",
    )

    # Add authentication middleware
    require_auth(app)

    # Load embeddings and determine model
    logger.info(f"Loading embeddings from {embeddings}...")
    device = get_device()
    logger.info(f"Using device: {device}")

    data = load_embeddings(embeddings, root=image_dir)
    image_paths = data["images"]
    image_embeddings = place(
        data["embeddings"], device, label=str(embeddings), scans_per_query=True
    )
    embedding_model_name = data.get("model", "clip")

    logger.info(f"Loaded {len(image_paths)} images")
    logger.info(f"Embedding model: {embedding_model_name}")

    # Load the embedder for text encoding
    embedder = get_embedder(embedding_model_name)
    embedder.load(device)

    supports_text_search = embedder.supports_text
    if not supports_text_search:
        logger.warning(
            f"Model '{embedding_model_name}' does not support text search. "
            "Text search will be disabled."
        )

    @app.route("/")
    def index():
        return render_template("index.html", supports_text_search=supports_text_search)

    @app.route("/search", methods=["POST"])
    def search_endpoint():
        prompt = request.form.get("prompt", "").strip()
        top_k = parse_top_k(request.form.get("top_k"))

        if top_k is None:
            return render_template(
                "results.html",
                error=TOP_K_ERROR,
                results=[],
                query_type="text",
                query_value="",
            )

        if not prompt:
            return render_template(
                "results.html",
                error="Please enter a search prompt",
                results=[],
                query_type="text",
                query_value="",
            )

        if not supports_text_search:
            return render_template(
                "results.html",
                error=f"Text search not supported with {embedding_model_name} embeddings. Use CLIP or Qwen3-VL.",
                results=[],
                query_type="text",
                query_value=prompt,
            )

        try:
            text_embedding = embedder.encode_text(prompt)
            top_indices, top_values = perform_search(
                top_k, text_embedding, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": display_name(image_paths[idx]),
                    }
                )

            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="text",
                query_value=prompt,
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return render_template(
                "results.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="text",
                query_value="",
            )

    @app.route("/search-image", methods=["POST"])
    def search_image_endpoint():
        """Search for similar images using an uploaded image"""
        top_k = parse_top_k(request.form.get("top_k"))
        if top_k is None:
            return render_template(
                "results.html",
                error=TOP_K_ERROR,
                results=[],
                query_type="image",
                query_value="",
            )

        # Check for file upload
        if "image" not in request.files:
            return render_template(
                "results.html",
                error="No image provided",
                results=[],
                query_type="image",
                query_value="",
            )

        file = request.files["image"]
        if file.filename == "":
            return render_template(
                "results.html",
                error="No image selected",
                results=[],
                query_type="image",
                query_value="",
            )

        try:
            # Load image from upload
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Encode the uploaded image
            query_embedding = embedder.encode_pil_image(pil_image)

            # Search for similar images
            top_indices, top_values = perform_search(
                top_k, query_embedding, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": display_name(image_paths[idx]),
                    }
                )

            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="image",
                query_value=file.filename or "uploaded image",
            )

        except Exception as e:
            logger.error(f"Image search error: {e}")
            return render_template(
                "results.html",
                error=f"Image search failed: {e!s}",
                results=[],
                query_type="image",
                query_value="",
            )

    @app.route("/similar/<int:index>")
    def similar_fullpage(index):
        """Find images similar to the image at the given index"""
        top_k = parse_top_k(request.args.get("top_k"))

        if index < 0 or index >= len(image_paths) or top_k is None:
            return render_template(
                "results.html",
                error="Image not found" if top_k is not None else TOP_K_ERROR,
                results=[],
                query_type="similar",
                query_value=index,
            )

        try:
            top_indices, top_values = find_similar(
                index, top_k, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": display_name(image_paths[idx]),
                    }
                )

            query_filename = display_name(image_paths[index])
            return render_template(
                "similar.html",
                results=results,
                error=None,
                query_type="similar",
                query_value=index,
                query_filename=query_filename,
            )

        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return render_template(
                "similar.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="similar",
                query_value=index,
            )

    @app.route("/api/similar/<int:index>")
    def similar_endpoint(index):
        """Find images similar to the image at the given index"""
        top_k = parse_top_k(request.args.get("top_k"))

        if index < 0 or index >= len(image_paths) or top_k is None:
            return render_template(
                "results.html",
                error="Image not found" if top_k is not None else TOP_K_ERROR,
                results=[],
                query_type="similar",
                query_value=index,
            )

        try:
            top_indices, top_values = find_similar(
                index, top_k, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": display_name(image_paths[idx]),
                    }
                )

            query_filename = display_name(image_paths[index])
            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="similar",
                query_value=index,
                query_filename=query_filename,
            )

        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return render_template(
                "results.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="similar",
                query_value=index,
            )

    @app.route("/image/<int:index>")
    def serve_image(index):
        """Serve image by index"""
        if index < 0 or index >= len(image_paths):
            return "Image not found", 404

        ref = image_paths[index]
        if is_url(ref):
            return redirect(ref)
        image_path = Path(ref)
        return send_from_directory(image_path.parent, image_path.name)

    run_app(app, host, port, open_browser=open_browser)
