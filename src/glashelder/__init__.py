"""Python CLI runs all of the glashelder experiments."""

import logging

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


@click.group()
def main():
    """Run the full glashelder CLI interface."""
    # load .env file
    load_dotenv()


def register_commands():
    # register cli commands defined in other files
    from metadata_extraction.cli import metadata_extraction

    from artmatcher.cli import artmatcher
    from classification.cli import classification
    from cluster.cli import cluster
    from embed.cli import embed
    from ocr.cli import ocr
    from preprocessing.cli import preprocessing
    from qwenvlm.cli import qwenvlm
    from risk_assessment.cli import risk_assessment

    main.add_command(cluster)
    main.add_command(qwenvlm)
    main.add_command(preprocessing)
    main.add_command(classification)
    main.add_command(risk_assessment)
    main.add_command(artmatcher)
    main.add_command(embed)
    main.add_command(ocr)
    main.add_command(metadata_extraction)


# Register commands on import so the CLI sees them.
register_commands()
