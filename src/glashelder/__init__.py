import click
from dotenv import load_dotenv
from pathlib import Path

# export the main CLI interface
__all__ = ["main"]

@click.group()
def main():
    """
    The main CLI interface
    """
    load_dotenv()
    pass


def register_commands():
    # register cli commands defined in other files
    from clipcluster.cli import clipcluster as clipcluster_cli
    
    main.add_command(clipcluster_cli)


register_commands()
