import glob
import logging

import click

from common import ASSETS_DIR, PROJECT_ROOT
from common.clipmodel import get_device

DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
PACKAGE_ROOT = PROJECT_ROOT / "embed"
EMBEDDINGS_PATH = PACKAGE_ROOT / "embeddings.pkl"

logger = logging.getLogger(__name__)


@click.group()
def embed():
    pass


@embed.command()
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--save-path", default=EMBEDDINGS_PATH, help="Embeddings pickle save path"
)
@click.option(
    "--count", default=10000, help="Amount of images to generate embeddings for"
)
@click.option("--batch-size", default=64, help="Batch size for embedding generation")
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use (clip, dinov2, or qwen3vl)",
)
def generate_embeddings(image_dir, save_path, count, batch_size, model):
    """Generate image embeddings using the specified model."""
    from embed import (
        get_embedder,
        save_embeddings,
    )

    device = get_device()

    logger.info(f"Loading embedding model: {model}")
    embedder = get_embedder(model)
    embedder.load(device)

    input_paths = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))[:count]
    logger.info(
        f"Generating embeddings for {len(input_paths)} images with batch size {batch_size}"
    )

    embeddings, processed_paths = embedder.encode_images(input_paths, batch_size)

    if len(processed_paths) < len(input_paths):
        logger.warning(
            f"Skipped {len(input_paths) - len(processed_paths)} images due to load errors"
        )

    save_embeddings(save_path, processed_paths, embeddings, embedder)
