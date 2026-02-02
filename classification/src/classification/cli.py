import click
import logging

from common import ASSETS_DIR

@click.group()
def classification():
    pass


@classification.command()
@click.option(
    "-i", "--image_dir",
    default=ASSETS_DIR,
    help="Directory where the images reside. Default is the assets directory."
)
@click.option(
    "-m", "--model",
    default="openai/clip-vit-large-patch14",
    help="CLIP model to use for classification. Default is 'openai/clip-vit-large-patch14'",
)
@click.option(
    "-b", "--max_batches",
    default=None,
    type=int,
    help="Maximum number of batches to process. Default is None (process all batches).",
)
@click.option(
    "-p", "--prompts",
    multiple=True,
    default=['a photo of a human'],
    help="Prompts to use for classification. Default is 'a photo of a human' You add multiple prompts by repeating the flag. Note that if you add only one prompt, this will result in all images being classified as that label.",
)
@click.option(
    "-l", "--labels",
    multiple=True,
    default=['human'],
    help="Labels corresponding to the prompts. Default is 'human'. You add multiple labels by repeating the flag.",
)
@click.option(
    "-c", "--confidence",
    default=0.4,
    type=float,
    help="Confidence threshold for classification. Default is 0.4",
)
@click.option(
    "-o", "--output_folder",
    required=True,
    help="Folder to save the classified images",
)


def classify_images(image_dir, model, max_batches, prompts, labels, confidence, output_folder=None):
    from .main import classify as cf
    cf(image_dir, model, max_batches, prompts, labels, confidence, output_folder)