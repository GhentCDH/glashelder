import click
from dotenv import load_dotenv

# export the main CLI interface
__all__ = ["main"]

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
    
    main.add_command(clipcluster)


if __name__ == "glashelder":
    register_commands()
