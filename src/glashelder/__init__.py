"""Python CLI runs all of the glashelder experiments."""

import logging
from importlib import import_module

import click
from dotenv import load_dotenv
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
    "classification": (
        "classification.cli:classification",
        "Classify images using CLIP zero-shot classification.",
    ),
    "cluster": (
        "cluster.cli:cluster",
        "Cluster and search images using embedding similarity.",
    ),
    "embed": ("embed.cli:embed", "Generate and manage image embeddings."),
    "ocr": ("ocr.cli:ocr", "Read text from images."),
    "preprocessing": (
        "preprocessing.cli:preprocessing",
        "Prepare datasets and train YOLO models for object detection.",
    ),
    "qwenvlm": (
        "qwenvlm.cli:qwenvlm",
        "Generate image descriptions using Qwen Vision-Language models.",
    ),
    "risk_assessment": (
        "risk_assessment.cli:risk_assessment",
        "Assess image content risk using vision-language models.",
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


@click.group(cls=LazyGroup)
def main():
    """Run the full glashelder CLI interface."""
    # load .env file
    load_dotenv()
