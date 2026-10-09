"""`glashelder setup`: a few plain-language questions that write `glashelder.toml`."""

import json
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

# torch's nvidia-cu13 wheels (from uv.lock after the pypi relock) need a driver
# that supports this CUDA version or newer; nvidia-smi's "CUDA Version" header
# is the max CUDA version that driver supports.
MIN_CUDA_VERSION = (13, 0)
README_GPU_FALLBACK = (
    "see the GPU section of README.md for how to pin an older CUDA build"
)


@dataclass
class Layout:
    """The paths setup derives from one data folder."""

    embeddings_glashelder: Path
    embeddings_reference: Path
    clusters: Path
    artworks: Path
    ml_model: Path
    image_cache: Path

    @classmethod
    def under(cls, data: Path) -> "Layout":
        return cls(
            embeddings_glashelder=data / "embeddings" / "glashelder-clip.store",
            embeddings_reference=data / "embeddings" / "artworks-clip.store",
            clusters=data / "clusters" / "glashelder-clip.clusters",
            artworks=data / "artworks.json.zst",
            ml_model=data / "models" / "artmatcher.pkl",
            image_cache=data / "image-cache",
        )


def existing_config(output: Path):
    """The config at `output`, or nothing, so a second run can default to the first's answers."""
    from common.config import Config

    if not output.is_file():
        return None
    try:
        return Config.load(output)
    except Exception:
        return None


def ask_images_folder(current: Path | None) -> Path:
    from common.images import list_images

    while True:
        answer = click.prompt(
            "Where are the glass plate photos?",
            default=str(current) if current else None,
            type=click.Path(path_type=Path),
        )
        folder = Path(answer).expanduser().resolve()
        if not folder.is_dir():
            console.print(f"[yellow]{folder} is not a folder.[/yellow]")
            continue
        count = len(list_images(folder))
        if count:
            console.print(f"found {count:,} images")
            return folder
        if click.confirm(f"Found 0 images in {folder}. Use it anyway?", default=False):
            return folder


def ask_data_folder(current: Path | None) -> Path:
    answer = click.prompt(
        "Where can glashelder keep the files it makes?",
        default=str(current) if current else "./data",
        type=click.Path(path_type=Path),
    )
    folder = Path(answer).expanduser().resolve()
    if not folder.is_dir() and click.confirm(f"Create {folder}?", default=True):
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def ask_customizations(layout: Layout) -> Layout:
    if not click.confirm("Change any of these locations?", default=False):
        return layout
    return Layout(
        embeddings_glashelder=Path(
            click.prompt(
                "Glashelder embeddings", default=str(layout.embeddings_glashelder)
            )
        ).expanduser(),
        embeddings_reference=Path(
            click.prompt(
                "Reference embeddings", default=str(layout.embeddings_reference)
            )
        ).expanduser(),
        clusters=Path(
            click.prompt("Clustering result", default=str(layout.clusters))
        ).expanduser(),
        artworks=Path(
            click.prompt("Scraped artworks index", default=str(layout.artworks))
        ).expanduser(),
        ml_model=Path(
            click.prompt("Trained matcher model", default=str(layout.ml_model))
        ).expanduser(),
        image_cache=Path(
            click.prompt("Image cache", default=str(layout.image_cache))
        ).expanduser(),
    )


def ask_secret() -> str:
    typed = click.prompt(
        "Password for the web pages (press Enter to make one)",
        default="",
        show_default=False,
        hide_input=True,
    )
    return typed or secrets.token_urlsafe(24)


def write_env_file(env_file: Path, secret: str) -> None:
    """Set `GLASHELDER_SECRET` in `env_file`, keeping every other line."""
    from common.fs import atomic_write

    lines = env_file.read_text().splitlines() if env_file.is_file() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith("GLASHELDER_SECRET="):
            lines[i] = f"GLASHELDER_SECRET={secret}"
            found = True
            break
    if not found:
        lines.append(f"GLASHELDER_SECRET={secret}")
    atomic_write(env_file, "\n".join(lines) + "\n")


def nvidia_smi_output(*args: str) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def check_gpu() -> None:
    if sys.platform == "darwin":
        return
    header = nvidia_smi_output()
    if header is None:
        console.print("No NVIDIA GPU found: models will run on the CPU, which is slow.")
        return

    name = nvidia_smi_output("--query-gpu=name", "--format=csv,noheader")
    name = name.strip().splitlines()[0] if name else "your GPU"

    match = re.search(r"CUDA Version:\s*([\d.]+)", header)
    if match is None:
        console.print(f"Found {name}, but could not read its driver's CUDA version.")
        return

    cuda_version = tuple(int(part) for part in match.group(1).split(".")[:2])
    if cuda_version >= MIN_CUDA_VERSION:
        console.print(f"Found {name}: the GPU will be used.")
    else:
        minimum = ".".join(str(part) for part in MIN_CUDA_VERSION)
        console.print(
            f"[yellow]Found {name}, but its driver only supports CUDA "
            f"{match.group(1)}; CUDA {minimum} or newer is needed. "
            f"Update the NVIDIA driver, or {README_GPU_FALLBACK}.[/yellow]"
        )


def toml_string(value: Path | str) -> str:
    return json.dumps(str(value))


def render_toml(images: Path, layout: Layout) -> str:
    return f"""\
# written by `glashelder setup`; every other setting and its default:
# glashelder.example.toml

[paths]
artworks = {toml_string(layout.artworks)}
ml_model = {toml_string(layout.ml_model)}
clusters = {toml_string(layout.clusters)}

[paths.images]
glashelder = {toml_string(images)}
cache = {toml_string(layout.image_cache)}

[paths.embeddings]
glashelder = {toml_string(layout.embeddings_glashelder)}
reference = {toml_string(layout.embeddings_reference)}
"""


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write glashelder.toml",
)
@click.option(
    "--env-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the web password",
)
def setup(output: Path | None, env_file: Path | None) -> None:
    """Answer a few questions and write glashelder.toml."""
    from common import PROJECT_ROOT
    from common.config import Config
    from common.fs import atomic_write

    output = output or PROJECT_ROOT / "glashelder.toml"
    env_file = env_file or PROJECT_ROOT / ".env"

    if output.is_file() and not click.confirm(
        f"{output} already exists. Overwrite it?", default=True
    ):
        return

    current = existing_config(output)
    current_images = current.paths.images.glashelder if current else None
    current_data = (
        current.paths.clusters.parent.parent
        if current and current.paths.clusters
        else None
    )

    images = ask_images_folder(current_images)
    data = ask_data_folder(current_data)
    layout = ask_customizations(Layout.under(data))

    for folder in (
        layout.embeddings_glashelder.parent,
        layout.embeddings_reference.parent,
        layout.clusters.parent,
        layout.ml_model.parent,
        layout.image_cache,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    secret = ask_secret()
    write_env_file(env_file, secret)

    atomic_write(output, render_toml(images, layout))
    Config.load(output)  # fails loudly if something above produced invalid toml

    check_gpu()

    table = Table(title="Written to " + str(output))
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("paths.images.glashelder", str(images))
    table.add_row("paths.embeddings.glashelder", str(layout.embeddings_glashelder))
    table.add_row("paths.embeddings.reference", str(layout.embeddings_reference))
    table.add_row("paths.clusters", str(layout.clusters))
    table.add_row("paths.artworks", str(layout.artworks))
    table.add_row("paths.ml_model", str(layout.ml_model))
    table.add_row("paths.images.cache", str(layout.image_cache))
    console.print(table)
    console.print(
        "\nNext steps:\n"
        "  uv run glashelder embed images\n"
        "  uv run glashelder cluster cluster\n"
        "  uv run glashelder cluster serve"
    )
