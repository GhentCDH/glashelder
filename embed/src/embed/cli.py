import glob
import logging
from pathlib import Path

import click

from common import ASSETS_DIR, PROJECT_ROOT, require_auth

from .embed import get_device

DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
PACKAGE_ROOT = PROJECT_ROOT / "embed"
EMBEDDINGS_PATH = PACKAGE_ROOT / "embeddings.pkl"

logger = logging.getLogger(__name__)


@click.group()
def embed():
    """Generate and manage image embeddings using various models."""
    pass


@embed.command()
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--artworks",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Scraped artworks (JSON or JSONL) to embed straight from their image URLs, "
    "instead of a local --image-dir",
)
@click.option(
    "--save-path", default=EMBEDDINGS_PATH, help="Embeddings pickle save path"
)
@click.option(
    "--count", default=10000, help="Amount of images to generate embeddings for"
)
@click.option("--batch-size", default=64, help="Batch size for embedding generation")
@click.option(
    "--concurrency",
    default=None,
    type=int,
    help="Image fetches in flight when embedding from URLs",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use (clip, dinov2, or qwen3vl)",
)
def generate_embeddings(
    image_dir, artworks, save_path, count, batch_size, concurrency, model
):
    """Generate image embeddings using the specified model.

    Images come either from a local directory (--image-dir) or from a scrape
    (--artworks), in which case they are fetched while the model works and never
    stored on disk.
    """
    from embed import (
        get_embedder,
        save_embeddings,
    )

    if artworks is None:
        assert image_dir is not None and Path(image_dir).is_dir(), (
            "The image directory (`--image-dir`) needs to exist and needs to be a directory!"
        )
        refs = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))[:count]
        root = image_dir
    else:
        import artscraper

        refs = [a.image_url for a in artscraper.read(artworks) if a.image_url][:count]
        root = None

    logger.info(
        f"Generating embeddings for {len(refs)} images, batch size {batch_size}"
    )

    device = get_device()

    logger.info(f"Loading embedding model: {model}")
    embedder = get_embedder(model)
    embedder.load(device)

    if artworks is None:
        embeddings, processed = embedder.encode_images(refs, batch_size)
    else:
        from embed.stream import encode_refs

        embeddings, processed = encode_refs(
            embedder,
            refs,
            batch_size=batch_size,
            concurrency=concurrency,
            on_batch=lambda done: logger.info(f"embedded {done}/{len(refs)}"),
        )

    if len(processed) < len(refs):
        logger.warning(
            f"Skipped {len(refs) - len(processed)} images due to load errors"
        )

    save_embeddings(save_path, processed, embeddings, embedder, root=root)


def perform_search(top_k, text_embedding, image_embeddings, image_paths):
    """Search for images similar to a text embedding."""
    import torch

    # Normalize embeddings
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    image_embeddings_norm = image_embeddings / image_embeddings.norm(
        dim=-1, keepdim=True
    )

    # Calculate cosine similarity
    similarities = (text_embedding @ image_embeddings_norm.T).squeeze(0)

    # Get top-k indices
    top_k = min(top_k, len(image_paths))
    top_values, top_indices = torch.topk(similarities, top_k)

    return top_indices, top_values


def find_similar(query_index, top_k, image_embeddings, image_paths, exclude_self=True):
    """Find images similar to a given image by index in the embedding space."""
    import torch

    query_embedding = image_embeddings[query_index].unsqueeze(0)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    image_embeddings_norm = image_embeddings / image_embeddings.norm(
        dim=-1, keepdim=True
    )
    similarities = (query_embedding @ image_embeddings_norm.T).squeeze(0)

    if exclude_self:
        similarities[query_index] = -float("inf")
        actual_top_k = min(top_k, len(image_paths) - 1)
    else:
        actual_top_k = min(top_k, len(image_paths))

    top_values, top_indices = torch.topk(similarities, actual_top_k)
    return top_indices, top_values


@embed.command()
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="Path to embeddings pickle file"
)
@click.option(
    "--image-dir",
    default=ASSETS_DIR,
    help="Directory the embedded images live in on this machine",
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
def serve(embeddings, image_dir, host, port):
    """Start a web server for interactive image search.

    Authentication is required. Set the GLASHELDER_SECRET environment variable,
    then access the server with /?secret=<your-secret> to get a session cookie.
    """
    import io

    from flask import Flask, render_template, request, send_from_directory
    from PIL import Image

    from embed import (
        get_embedder,
        load_embeddings,
    )

    app = Flask(
        __name__,
        template_folder=PACKAGE_ROOT / "templates",
        static_folder=PACKAGE_ROOT / "static",
    )

    # Add authentication middleware
    require_auth(app)

    # Load embeddings and determine model
    logger.info(f"Loading embeddings from {embeddings}...")
    device = get_device()
    logger.info(f"Using device: {device}")

    data = load_embeddings(embeddings, root=image_dir)
    image_paths = data["images"]
    image_embeddings = data["embeddings"].to(device)
    embedding_model_name = data.get("model", "clip")

    logger.info(f"Loaded {len(image_paths)} images")
    logger.info(f"Embedding model: {embedding_model_name}")

    # Load the embedder for text encoding
    embedder = get_embedder(embedding_model_name)
    embedder.load(device)

    supports_text_search = embedder.supports_text
    if not supports_text_search:
        logger.warning(
            f"Model '{embedding_model_name}' does not support text search. "
            "Text search will be disabled."
        )

    @app.route("/")
    def index():
        return render_template("index.html", supports_text_search=supports_text_search)

    @app.route("/search", methods=["POST"])
    def search_endpoint():
        prompt = request.form.get("prompt", "").strip()
        top_k = int(request.form.get("top_k", 12))

        if not prompt:
            return render_template(
                "results.html",
                error="Please enter a search prompt",
                results=[],
                query_type="text",
                query_value="",
            )

        if not supports_text_search:
            return render_template(
                "results.html",
                error=f"Text search not supported with {embedding_model_name} embeddings. Use CLIP or Qwen3-VL.",
                results=[],
                query_type="text",
                query_value=prompt,
            )

        try:
            text_embedding = embedder.encode_text(prompt)
            top_indices, top_values = perform_search(
                top_k, text_embedding, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": Path(image_paths[idx]).name,
                    }
                )

            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="text",
                query_value=prompt,
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return render_template(
                "results.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="text",
                query_value="",
            )

    @app.route("/search-image", methods=["POST"])
    def search_image_endpoint():
        """Search for similar images using an uploaded image"""
        top_k = int(request.form.get("top_k", 12))

        # Check for file upload
        if "image" not in request.files:
            return render_template(
                "results.html",
                error="No image provided",
                results=[],
                query_type="image",
                query_value="",
            )

        file = request.files["image"]
        if file.filename == "":
            return render_template(
                "results.html",
                error="No image selected",
                results=[],
                query_type="image",
                query_value="",
            )

        try:
            # Load image from upload
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Encode the uploaded image
            query_embedding = embedder.encode_pil_image(pil_image)

            # Search for similar images
            top_indices, top_values = perform_search(
                top_k, query_embedding, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": Path(image_paths[idx]).name,
                    }
                )

            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="image",
                query_value=file.filename or "uploaded image",
            )

        except Exception as e:
            logger.error(f"Image search error: {e}")
            return render_template(
                "results.html",
                error=f"Image search failed: {e!s}",
                results=[],
                query_type="image",
                query_value="",
            )

    @app.route("/similar/<int:index>")
    def similar_fullpage(index):
        """Find images similar to the image at the given index"""
        top_k = int(request.args.get("top_k", 12))

        if index < 0 or index >= len(image_paths):
            return render_template(
                "results.html",
                error="Image not found",
                results=[],
                query_type="similar",
                query_value=index,
            )

        try:
            top_indices, top_values = find_similar(
                index, top_k, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": Path(image_paths[idx]).name,
                    }
                )

            query_filename = Path(image_paths[index]).name
            return render_template(
                "similar.html",
                results=results,
                error=None,
                query_type="similar",
                query_value=index,
                query_filename=query_filename,
            )

        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return render_template(
                "similar.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="similar",
                query_value=index,
            )

    @app.route("/api/similar/<int:index>")
    def similar_endpoint(index):
        """Find images similar to the image at the given index"""
        top_k = int(request.args.get("top_k", 12))

        if index < 0 or index >= len(image_paths):
            return render_template(
                "results.html",
                error="Image not found",
                results=[],
                query_type="similar",
                query_value=index,
            )

        try:
            top_indices, top_values = find_similar(
                index, top_k, image_embeddings, image_paths
            )

            results = []
            for rank, (idx, score) in enumerate(
                zip(top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False),
                1,
            ):
                results.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "score": float(score),
                        "filename": Path(image_paths[idx]).name,
                    }
                )

            query_filename = Path(image_paths[index]).name
            return render_template(
                "results.html",
                results=results,
                error=None,
                query_type="similar",
                query_value=index,
                query_filename=query_filename,
            )

        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return render_template(
                "results.html",
                error=f"Search failed: {e!s}",
                results=[],
                query_type="similar",
                query_value=index,
            )

    @app.route("/image/<int:index>")
    def serve_image(index):
        """Serve image by index"""
        if index < 0 or index >= len(image_paths):
            return "Image not found", 404

        image_path = Path(image_paths[index])
        return send_from_directory(image_path.parent, image_path.name)

    logger.info(f"Starting Flask server on {host}:{port}...")
    app.run(host=host, port=port, debug=True)
