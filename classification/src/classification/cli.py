import click

IMAGE_DIR = "/mnt/UGent_Share/glashelder/assets"

from .main import classify as cf


@click.group()
def classification():
    pass


@classification.command()
@click.option(
    "--image_dir", default=IMAGE_DIR, help="Directory where the images reside"
)
def classify(image_dir):
    cf(image_dir)
