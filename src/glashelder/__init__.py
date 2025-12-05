import logging

from rich.logging import RichHandler
from rich.traceback import install
import click
from dotenv import load_dotenv

# export the main CLI interface
__all__ = ["main"]

install(show_locals=True)
FORMAT="%(message)s"
DATE_FORMAT="[%X]"

logging.basicConfig(level="INFO", format=FORMAT, datefmt=DATE_FORMAT, handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True,show_time=True, show_level=True, show_path=True)])

@click.group()
def main():
    """
    The main CLI interface
    """

    # load .env file
    load_dotenv()


def register_commands():
    # register cli commands defined in other files
    from clipcluster.cli import clipcluster
    from qwenvlm.cli import qwenvlm
    from preprocessing.cli import preprocessing
    from classification.cli import classification

    main.add_command(clipcluster)
    main.add_command(qwenvlm)
    main.add_command(preprocessing)
    main.add_command(classification)


if __name__ == "glashelder":
    register_commands()
