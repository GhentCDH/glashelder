"""The image search viewer: text and image queries against an embedding store."""

import io
import logging
from pathlib import Path

from flask import Flask, redirect, render_template, request, send_from_directory
from PIL import Image

from common.images import is_url

from .cli import TOP_K_ERROR, display_name, parse_top_k
from .device import get_device
from .io import load_embeddings
from .models import get_embedder
from .search import find_similar, perform_search, place

logger = logging.getLogger(__name__)


def create_app(embeddings, image_dir) -> Flask:
    app = Flask(
        __name__,
        template_folder=Path(__file__).parent / "templates",
        static_folder=Path(__file__).parent / "static",
    )

    logger.info(f"Loading embeddings from {embeddings}...")
    device = get_device()
    logger.info(f"Using device: {device}")

    data = load_embeddings(embeddings, root=image_dir)
    image_paths = data["images"]
    image_embeddings = place(
        data["embeddings"], device, label=str(embeddings), scans_per_query=True
    )
    embedding_model_name = data.get("model", "clip")

    logger.info(f"Loaded {len(image_paths)} images")
    logger.info(f"Embedding model: {embedding_model_name}")

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
        top_k = parse_top_k(request.form.get("top_k"))

        if top_k is None:
            return render_template(
                "results.html",
                error=TOP_K_ERROR,
                results=[],
                query_type="text",
                query_value="",
            )

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
                error=f"Text search not supported with {embedding_model_name} embeddings. Use CLIP.",
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
                        "filename": display_name(image_paths[idx]),
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
        top_k = parse_top_k(request.form.get("top_k"))
        if top_k is None:
            return render_template(
                "results.html",
                error=TOP_K_ERROR,
                results=[],
                query_type="image",
                query_value="",
            )

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
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            query_embedding = embedder.encode_pil_image(pil_image)

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
                        "filename": display_name(image_paths[idx]),
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
        top_k = parse_top_k(request.args.get("top_k"))

        if index < 0 or index >= len(image_paths) or top_k is None:
            return render_template(
                "results.html",
                error="Image not found" if top_k is not None else TOP_K_ERROR,
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
                        "filename": display_name(image_paths[idx]),
                    }
                )

            query_filename = display_name(image_paths[index])
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
        top_k = parse_top_k(request.args.get("top_k"))

        if index < 0 or index >= len(image_paths) or top_k is None:
            return render_template(
                "results.html",
                error="Image not found" if top_k is not None else TOP_K_ERROR,
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
                        "filename": display_name(image_paths[idx]),
                    }
                )

            query_filename = display_name(image_paths[index])
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

        ref = image_paths[index]
        if is_url(ref):
            return redirect(ref)
        image_path = Path(ref)
        return send_from_directory(image_path.parent, image_path.name)

    return app
