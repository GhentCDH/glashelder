import click

from .main import classify as cf
from common import ASSETS_DIR


@click.group()
def classification():
    pass


@classification.command()
@click.option(
    "--image_dir", default=ASSETS_DIR, help="Directory where the images reside"
)
def classify(image_dir):
    cf(image_dir)
