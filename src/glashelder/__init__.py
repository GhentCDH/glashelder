"""Python CLI runs all of the glashelder experiments."""

import logging

import click
from dotenv import load_dotenv
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
    from risk_assessment.cli import risk_assessment

    from classification.cli import classification
    from clipcluster.cli import clipcluster
    from preprocessing.cli import preprocessing

    main.add_command(clipcluster)
    main.add_command(preprocessing)
    main.add_command(classification)
    main.add_command(risk_assessment)


# Register commands on import so the CLI sees them.
register_commands()
