"""Python CLI runs all of the glashelder experiments."""

import logging
from importlib import import_module
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# export the main CLI interface
__all__ = ["main"]

install(show_locals=True)
FORMAT = "%(message)s"
DATE_FORMAT = "[%X]"

logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_level=True,
            show_path=True,
        )
    ],
)


# Command groups, kept as import paths rather than imports: several packages pull
# in torch and transformers at import time, which costs hundreds of megabytes and
# seconds of startup for every command, including the ones that never touch a model.
COMMANDS: dict[str, tuple[str, str]] = {
    "artmatcher": (
        "artmatcher.cli:artmatcher",
        "Match glass plate photos to known artworks.",
    ),
    "artscraper": (
        "artscraper.cli:artscraper",
        "Scrape an index of artworks from external data sources.",
    ),
    "cluster": (
        "cluster.cli:cluster",
        "Cluster images using embedding similarity.",
    ),
    "config": (
        "common.config_example:config",
        "Show the settings glashelder.toml can hold.",
    ),
    "embed": ("embed.cli:embed", "Generate and manage image embeddings."),
    "ocr": ("ocr.cli:ocr", "Read text from images."),
    "preprocessing": (
        "preprocessing.cli:preprocessing",
        "Prepare datasets and train YOLO models for object detection.",
    ),
    "glaswacht": (
        "glaswacht.cli:glaswacht",
        "Assess image content risk using vision-language models.",
    ),
    "setup": (
        "glashelder.setup:setup",
        "Answer a few questions and write glashelder.toml.",
    ),
}


class LazyGroup(click.Group):
    """A group that imports a subcommand's package only when it is used."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(COMMANDS)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name not in COMMANDS:
            return super().get_command(ctx, name)
        path, _ = COMMANDS[name]
        module, attribute = path.split(":")
        return getattr(import_module(module), attribute)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # from the table above, so `--help` does not import every package either
        with formatter.section("Commands"):
            formatter.write_dl(
                [(name, help) for name, (_, help) in sorted(COMMANDS.items())]
            )


def cli_defaults() -> dict:
    """Every command's defaults from `glashelder.toml`, as click's nested `default_map`.

    Only settings that are set are handed over, so an unset one keeps the default
    declared on the command itself. `common.config` imports nothing heavy, so this
    costs nothing for commands that do not use it.
    """
    from common.config import config, defaults_for

    settings = config()
    paths = settings.paths

    def given(**values) -> dict:
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in values.items()
            if value is not None
        }

    matcher = defaults_for(settings.artmatcher)
    workers = matcher.pop("workers", None)
    matching = {
        **matcher,
        **given(
            artworks=paths.artworks,
            glashelder_images=paths.images.glashelder,
            glashelder_embeddings=paths.embeddings.glashelder,
            reference_embeddings=paths.embeddings.reference,
            reference_images=paths.images.reference,
            ml_model=paths.ml_model,
        ),
    }
    embedding = defaults_for(settings.embed)
    ocr_options = {k: v for k, v in defaults_for(settings.ocr).items() if k != "cache"}
    ocr_no_concurrency = {k: v for k, v in ocr_options.items() if k != "concurrency"}
    return {
        "artscraper": {
            "scrape": given(output=paths.artworks),
        },
        "cluster": {
            "cluster": {
                **defaults_for(settings.cluster),
                **given(
                    embeddings=paths.embeddings.glashelder,
                    image_dir=paths.images.glashelder,
                    output=paths.clusters,
                ),
            },
            "info": given(path=paths.clusters),
            "serve": given(clusters=paths.clusters, image_dir=paths.images.glashelder),
            "report": given(path=paths.clusters, image_dir=paths.images.glashelder),
            "analyze": given(
                embeddings=paths.embeddings.glashelder,
                image_dir=paths.images.glashelder,
            ),
        },
        "embed": {
            "artworks": {
                **embedding,
                **given(artworks=paths.artworks, output=paths.embeddings.reference),
            },
            "images": {
                **embedding,
                **given(
                    image_dir=paths.images.glashelder,
                    output=paths.embeddings.glashelder,
                ),
            },
            "neighbours": given(
                queries=paths.embeddings.glashelder,
                references=paths.embeddings.reference,
                top_k=settings.artmatcher.top_k,
            ),
            "search": given(
                embeddings=paths.embeddings.glashelder,
                image_dir=paths.images.glashelder,
            ),
            "serve": given(
                embeddings=paths.embeddings.glashelder,
                image_dir=paths.images.glashelder,
            ),
        },
        "artmatcher": {
            "serve": {**matching, **given(search_workers=workers)},
            "batch-classify": {**matching, **given(workers=workers)},
            "train-classifier": given(workers=workers),
            "report": given(artworks=paths.artworks),
        },
        "glaswacht": {
            "serve": given(
                csv_path=settings.glaswacht.results,
                images_dir=settings.glaswacht.images,
            ),
        },
        "ocr": {
            "one": ocr_no_concurrency,
            "many": ocr_options,
            "annotate": {**ocr_no_concurrency, **given(images=paths.images.glashelder)},
            "validate": ocr_options,
        },
    }


@click.group(cls=LazyGroup)
@click.pass_context
def main(ctx: click.Context):
    """Run the full glashelder CLI interface."""
    if ctx.invoked_subcommand in ("setup", "config"):
        # a broken or missing toml must not block the commands that write one
        return
    from common.config import find_config

    if find_config() is None:
        logging.getLogger(__name__).info(
            "no glashelder.toml yet, run `uv run glashelder setup`"
        )
    ctx.default_map = cli_defaults()
