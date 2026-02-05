import glob
import logging
import os
import shutil
from pathlib import Path

# installed packages
import click
from dotenv import load_dotenv
from PIL import Image

# local source
from common import ASSETS_DIR, PROJECT_ROOT
from common.clipmodel import get_device

from .embed import (
    CLIP_MODEL_ID,
    get_embedder,
    load_embeddings,
    save_embeddings,
)

# TODO: put some default images in `./images/` as exampels
DEFAULT_IMAGES = "images/"
CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR = "clustered_samples"
PACKAGE_ROOT = PROJECT_ROOT / "clipcluster"
EMBEDDINGS_PATH = PACKAGE_ROOT / "embeddings.pkl"

logger = logging.getLogger(__name__)


def resize_image_to_max_dimension(src_path: str, dst_path: str, max_dim: int = 512):
    """
    Resize an image to fit within max_dim x max_dim while preserving aspect ratio.

    Args:
        src_path: Source image path
        dst_path: Destination image path
        max_dim: Maximum width or height in pixels
    """
    img = Image.open(src_path)
    quality = 75

    # Calculate new dimensions while preserving aspect ratio
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        # Image is already small enough, just copy it
        img.save(dst_path, quality=quality)
        return

    # Calculate scaling factor
    scale = min(max_dim / width, max_dim / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize using high-quality resampling
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_img.save(dst_path, quality=quality)


def perform_search(top_k, text_embedding, image_embeddings, image_paths):
    """Shared search logic"""
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
    """Find images similar to a given image by index in the embedding space"""
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


@click.group()
def clipcluster():
    load_dotenv()
    pass


@clipcluster.command()
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="The path to a embeddings pickle file"
)
@click.option("--sample-size", default=100000, help="The amount of images to cluster")
@click.option(
    "--batch-size", default=64, help="The batch size for embeddings generation"
)
@click.option(
    "-o",
    "--output-files",
    is_flag=True,
    help="Whether or not to write example images to files",
)
@click.option("--count", default=40000, help="How many points to log to rerun")
@click.option(
    "-r",
    "--reduce",
    is_flag=True,
    help="Whether or not to stop after reducing dimensions",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use if generating embeddings",
)
def cluster(
    image_dir, embeddings, sample_size, batch_size, output_files, count, reduce, model
):
    from .cluster import create_clusterer, reduce_dimensions
    from .rr import init_rerun, log_clustered_points, log_images_3d, log_reduced

    device = get_device()

    init_rerun(os.getenv("RERUN_ADDR"))

    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))

    if embeddings is not None and os.path.exists(embeddings):
        logger.info("Loading embeddings")
        data = load_embeddings(embeddings)
        image_paths = data["images"]
        embeddings = data["embeddings"].to(device)
        logger.info(f"Loaded embeddings from model: {data.get('model', 'unknown')}")
    else:
        logger.info(f"Getting embedding model: {model}")
        embedder = get_embedder(model)
        embedder.load(device)
        # Get all image paths
        input_paths = list(glob.glob(f"{image_dir}/*.jpg"))[:sample_size]
        logger.info("Generating image embeddings...")
        embeddings, image_paths = embedder.encode_images(input_paths, batch_size)
        if len(image_paths) < len(input_paths):
            logger.warning(
                f"Skipped {len(input_paths) - len(image_paths)} images due to load errors"
            )

    logger.info("Reducing dimensions")
    # reduced_embeddings = embeddings.cpu().numpy()
    reduced_embeddings = reduce_dimensions(embeddings.cpu().numpy(), 30)
    visualization_embeddings = reduce_dimensions(embeddings.cpu().numpy(), 3)

    if reduce:
        log_reduced(visualization_embeddings, count=20000)
        return

    logger.info("cluster points")
    clusterer = create_clusterer(reduced_embeddings)

    # log to rerun
    logger.info("logging clustered points to rerun")
    log_clustered_points(visualization_embeddings, clusterer.labels_, count=count)
    log_images_3d(visualization_embeddings, image_paths, 300)

    # Get cluster labels
    labels = clusterer.labels_

    # Print cluster statistics
    unique_labels = set(labels)
    logger.info(
        f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
    )
    logger.info(f"Number of noise points: {sum(labels == -1)}")
    logger.info("Cluster sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            logger.info(f"  Noise: {sum(labels == label)} points")
        else:
            logger.info(f"  Cluster {label}: {sum(labels == label)} points")

    if not output_files:
        return

    # Create output directory
    output_dir = Path(CLUSTER_IMAGE_SAMPLES_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Get unique clusters (excluding noise)
    unique_clusters = sorted([label for label in set(labels) if label != -1])

    # For each cluster, copy 3 sample images
    for cluster_id in unique_clusters:
        # Create cluster directory
        cluster_dir = output_dir / f"cluster_{cluster_id}"
        # remove the directory if it already exists
        shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(exist_ok=True)

        # Get indices of images in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

        # Select up to 2 samples
        sample_indices = cluster_indices[:4]

        # Copy and resize images
        for idx in sample_indices:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = cluster_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        print(
            f"Cluster {cluster_id}: {len(cluster_indices)} total images, saved {len(sample_indices)} samples"
        )

    # Optionally handle noise points
    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    if noise_indices:
        noise_dir = output_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        # Save up to 2 noise samples
        for idx in noise_indices[:2]:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = noise_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        print(
            f"Noise: {len(noise_indices)} total images, saved {min(2, len(noise_indices))} samples"
        )

    print(f"\nSamples saved to: {output_dir}")


@clipcluster.command()
@click.argument("prompt")
@click.option("--top-k", default=10, help="Number of top matching images to return")
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--output-dir", default="search_results", help="Directory to save matching images"
)
@click.option(
    "--sample-size",
    default=10000,
    help="The amount of images in the dataset to consider",
)
@click.option("--embeddings", default=None, help="The path to a embeddings pickle file")
@click.option("--batch-size", default=64, help="Batch size for embedding generation")
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use",
)
def search(
    prompt, top_k, image_dir, output_dir, sample_size, embeddings, batch_size, model
):
    """Search for images matching a text prompt using embedding similarity"""
    logger.info(f"Searching for: '{prompt}'")
    logger.info(f"Looking for top {top_k} matches")

    device = get_device()

    # Get image embeddings and determine which model to use for text encoding
    if embeddings is not None:
        data = load_embeddings(embeddings)
        image_paths = data["images"]
        image_embeddings = data["embeddings"].to(device)
        embedding_model_name = data.get("model", "clip")
        logger.info(f"Loaded embeddings from model: {embedding_model_name}")
    else:
        embedding_model_name = model
        logger.info(f"Getting embedding model: {model}")
        embedder = get_embedder(model)
        embedder.load(device)
        input_paths = list(glob.glob(f"{image_dir}/*.jpg"))[:sample_size]
        logger.info("Generating image embeddings...")
        image_embeddings, image_paths = embedder.encode_images(input_paths, batch_size)
        if len(image_paths) < len(input_paths):
            logger.warning(
                f"Skipped {len(input_paths) - len(image_paths)} images due to load errors"
            )

    # Load the embedder for text encoding (must match the image embeddings)
    embedder = get_embedder(embedding_model_name)
    embedder.load(device)

    if not embedder.supports_text:
        logger.error(
            f"Model '{embedding_model_name}' does not support text search. "
            "Use CLIP or Qwen3-VL for text-based search."
        )
        return

    # Get text embedding for the prompt
    logger.info("Generating text embedding...")
    text_embedding = embedder.encode_text(prompt)

    top_indices, top_scores = perform_search(
        top_k, text_embedding, image_embeddings, image_paths
    )

    # Print results
    logger.info(f"\nTop {top_k} matches for '{prompt}':")
    for i, (idx, score) in enumerate(
        zip(top_indices.cpu().numpy(), top_scores.cpu().numpy(), strict=False), 1
    ):
        logger.info(f"{i}. {Path(image_paths[idx]).name} (similarity: {score:.4f})")

    # Create output directory and copy images
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Clear previous results
    for old_file in output_path.glob("*"):
        old_file.unlink()

    for i, (idx, score) in enumerate(
        zip(top_indices.cpu().numpy(), top_scores.cpu().numpy(), strict=False), 1
    ):
        src_path = image_paths[idx]
        filename = Path(src_path).name
        # Add rank and score to filename
        dst_filename = f"{i:02d}_score{score:.3f}_{filename}"
        dst_path = output_path / dst_filename
        shutil.copy(src_path, dst_path)

    logger.info(f"\nTop {top_k} images saved to: {output_path}")


@clipcluster.command()
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
    device = get_device()

    logger.info(f"Loading embedding model: {model}")
    embedder = get_embedder(model)
    embedder.load(device)

    input_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))[:count]
    logger.info(
        f"Generating embeddings for {len(input_paths)} images with batch size {batch_size}"
    )

    embeddings, processed_paths = embedder.encode_images(input_paths, batch_size)

    if len(processed_paths) < len(input_paths):
        logger.warning(
            f"Skipped {len(input_paths) - len(processed_paths)} images due to load errors"
        )

    save_embeddings(save_path, processed_paths, embeddings, embedder)


@clipcluster.command()
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="Path to embeddings pickle file"
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
def serve(embeddings, host, port):
    """Start a web server for interactive image search"""
    import io

    from flask import Flask, render_template, request, send_from_directory

    app = Flask(
        __name__,
        template_folder=PACKAGE_ROOT / "templates",
        static_folder=PACKAGE_ROOT / "static",
    )

    # Load embeddings and determine model
    logger.info(f"Loading embeddings from {embeddings}...")
    device = get_device()
    logger.info(f"Using device: {device}")

    data = load_embeddings(embeddings)
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


@clipcluster.command()
@click.option(
    "--image-dir", default=ASSETS_DIR, help="Directory containing images to search"
)
@click.option(
    "--embeddings", default=EMBEDDINGS_PATH, help="Path to embeddings pickle file"
)
@click.option("--sample-size", default=100000, help="Amount of images to cluster")
@click.option("--batch-size", default=64, help="Batch size for embeddings generation")
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save analysis results (default: cluster_analysis_YYYYMMDD_HHMMSS)",
)
@click.option("--min-cluster-size", default=50, help="Minimum cluster size for HDBSCAN")
@click.option(
    "--hierarchical",
    is_flag=True,
    help="Enable hierarchical refinement of vague clusters",
)
@click.option(
    "--density-threshold",
    default=1.0,
    help="Density threshold for identifying vague clusters",
)
@click.option("--max-depth", default=2, help="Maximum recursion depth for refinement")
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use if generating embeddings",
)
def analyze(
    image_dir,
    embeddings,
    sample_size,
    batch_size,
    output_dir,
    min_cluster_size,
    hierarchical,
    density_threshold,
    max_depth,
    model,
):
    """Analyze clustering results and generate statistics and visualizations."""
    from datetime import datetime

    from .cluster import (
        calculate_cluster_statistics,
        create_2d_visualization,
        create_3d_visualization,
        create_clusterer,
        reduce_dimensions,
        save_cluster_statistics,
    )
    from .rr import init_rerun, log_clustered_points_density

    init_rerun(os.getenv("RERUN_ADDR"))

    device = get_device()

    # Add timestamp to output directory if not explicitly specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"cluster_analysis_{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load or generate embeddings
    if embeddings is not None and os.path.exists(embeddings):
        logger.info("Loading embeddings")
        data = load_embeddings(embeddings)
        image_paths = data["images"][:sample_size]
        embeddings_tensor = data["embeddings"][:sample_size].to(device)
        embedding_model_name = data.get("model", "clip")
        embedding_model_id = data.get("model_id", CLIP_MODEL_ID)
        logger.info(f"Loaded embeddings from model: {embedding_model_name}")
    else:
        logger.info(f"Getting embedding model: {model}")
        embedder = get_embedder(model)
        embedder.load(device)
        embedding_model_name = embedder.name
        embedding_model_id = embedder.model_id
        input_paths = list(glob.glob(f"{image_dir}/*.jpg"))[:sample_size]
        logger.info("Generating image embeddings...")
        embeddings_tensor, image_paths = embedder.encode_images(input_paths, batch_size)
        if len(image_paths) < len(input_paths):
            logger.warning(
                f"Skipped {len(input_paths) - len(image_paths)} images due to load errors"
            )

    logger.info(f"Processing {len(image_paths)} images")

    # Reduce dimensions for clustering (30D) and visualization (2D and 3D)
    logger.info("Reducing dimensions for clustering (30D)...")
    reduced_embeddings_30d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 30)

    logger.info("Reducing dimensions for 2D visualization...")
    reduced_embeddings_2d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 2)

    logger.info("Reducing dimensions for 3D visualization...")
    reduced_embeddings_3d = reduce_dimensions(embeddings_tensor.cpu().numpy(), 3)

    # Perform clustering
    logger.info(f"Clustering with min_cluster_size={min_cluster_size}...")
    clusterer = create_clusterer(
        reduced_embeddings_30d, min_cluster_size=min_cluster_size
    )
    labels = clusterer.labels_

    # Apply hierarchical refinement if enabled
    if hierarchical:
        from .cluster import HierarchicalConfig, hierarchical_cluster

        config = HierarchicalConfig(
            enabled=True,
            density_threshold=density_threshold,
            max_depth=max_depth,
            min_cluster_size=min_cluster_size,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Starting hierarchical refinement...")
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  Density threshold: {density_threshold}")
        logger.info(f"  Max depth: {max_depth}")
        logger.info(f"  Min cluster size: {min_cluster_size}")

        labels, reduced_embeddings_30d, refinement_tree = hierarchical_cluster(
            embeddings_tensor.cpu().numpy(), labels, reduced_embeddings_30d, config
        )

    # Calculate statistics
    logger.info("Calculating cluster statistics...")
    stats = calculate_cluster_statistics(reduced_embeddings_30d, labels, clusterer)

    # log to rerun
    log_clustered_points_density(
        reduced_embeddings_3d, labels, stats, 30000, "clusters"
    )

    # Add hierarchical metadata if used
    if hierarchical:
        stats["refinement_tree"] = refinement_tree
        stats["metadata_hierarchical"] = {
            "enabled": True,
            "density_threshold": density_threshold,
            "max_depth": max_depth,
        }

    # Define output paths
    json_path = output_path / "cluster_stats.json"
    plot_2d_path = output_path / "clusters_2d.png"
    plot_3d_path = output_path / "clusters_3d.png"

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    create_2d_visualization(reduced_embeddings_2d, labels, str(plot_2d_path))
    create_3d_visualization(reduced_embeddings_3d, labels, str(plot_3d_path))

    # Save sample images for each cluster
    logger.info("Saving sample images for each cluster...")
    samples_dir = output_path / "cluster_samples"
    samples_dir.mkdir(exist_ok=True)

    # Convert labels to strings for consistent comparison
    labels_str = [str(label) for label in labels]

    # Get unique clusters (excluding noise and .noise suffix)
    unique_clusters = sorted(
        [
            label
            for label in set(labels_str)
            if label != "-1" and not label.endswith(".noise")
        ],
        key=lambda x: (x.count("."), x),
    )  # Sort by depth, then name

    # For each cluster, copy sample images
    for cluster_id in unique_clusters:
        # Create cluster directory (handle hierarchical labels with dots)
        cluster_dir = samples_dir / f"cluster_{cluster_id}"
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(exist_ok=True, parents=True)

        # Get indices of images in this cluster
        cluster_indices = [
            i for i, label in enumerate(labels_str) if label == cluster_id
        ]

        # Select up to 4 samples
        sample_indices = cluster_indices[:4]

        # Copy and resize images
        for idx in sample_indices:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = cluster_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        logger.info(
            f"Cluster {cluster_id}: {len(cluster_indices)} total images, saved {len(sample_indices)} samples"
        )

    # Handle noise points (both "-1" and ".noise" suffix)
    noise_indices = [
        i
        for i, label in enumerate(labels_str)
        if label == "-1" or label.endswith(".noise")
    ]
    if noise_indices:
        noise_dir = samples_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        # Save up to 2 noise samples
        for idx in noise_indices[:2]:
            src_path = image_paths[idx]
            filename = Path(src_path).name
            dst_path = noise_dir / filename
            resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

        logger.info(
            f"Noise: {len(noise_indices)} total images, saved {min(2, len(noise_indices))} samples"
        )

    # Also save hierarchical .noise clusters separately
    if hierarchical:
        noise_clusters = sorted(
            [label for label in set(labels_str) if label.endswith(".noise")]
        )
        for noise_cluster_id in noise_clusters:
            noise_cluster_dir = samples_dir / f"cluster_{noise_cluster_id}"
            noise_cluster_dir.mkdir(exist_ok=True, parents=True)

            noise_cluster_indices = [
                i for i, label in enumerate(labels_str) if label == noise_cluster_id
            ]
            for idx in noise_cluster_indices[:4]:
                src_path = image_paths[idx]
                filename = Path(src_path).name
                dst_path = noise_cluster_dir / filename
                resize_image_to_max_dimension(src_path, str(dst_path), max_dim=512)

            logger.info(
                f"Cluster {noise_cluster_id}: {len(noise_cluster_indices)} total images, saved {min(4, len(noise_cluster_indices))} samples"
            )

    # Add metadata to stats (including visualization paths)
    stats["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(image_paths),
        "min_cluster_size": min_cluster_size,
        "embedding_model": embedding_model_name,
        "embedding_model_id": embedding_model_id,
        "reduction_dims": 30,
        "output_directory": str(output_path.resolve()),
        "visualizations": {
            "plot_2d": str(plot_2d_path.resolve()),
            "plot_3d": str(plot_3d_path.resolve()),
        },
        "sample_images": str(samples_dir.resolve()),
    }

    # Print hierarchical refinement summary if applicable
    if hierarchical and refinement_tree:
        logger.info("\n" + "=" * 60)
        logger.info("HIERARCHICAL CLUSTERING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Refinement depth: {max_depth}")
        logger.info(f"Clusters refined: {len(refinement_tree)}")

        total_sub_clusters = 0
        for parent_id, info in sorted(
            refinement_tree.items(), key=lambda x: (x[0].count("."), x[0])
        ):
            sub_cluster_count = len(
                [c for c in info["refined_into"] if not c.endswith(".noise")]
            )
            total_sub_clusters += sub_cluster_count

            logger.info(f"\nCluster {parent_id} (depth {info['depth']}):")
            logger.info(
                f"  Original: {info['original_size']} pts, density {info['original_density']:.2f}"
            )
            logger.info(f"  Refined into {len(info['refined_into'])} sub-clusters:")

            # Show breakdown of sub-clusters with their stats
            for sub_label in sorted(
                info["refined_into"], key=lambda x: (x.count("."), x)
            ):
                # Find stats for this sub-cluster
                sub_stats = next(
                    (c for c in stats["clusters"] if str(c["cluster_id"]) == sub_label),
                    None,
                )
                if sub_stats:
                    status = (
                        "✓"
                        if sub_stats["density"] >= density_threshold
                        else "→ needs refinement"
                    )
                    logger.info(
                        f"    ├─ {sub_label}: {sub_stats['size']} pts "
                        f"(density {sub_stats['density']:.2f}) {status}"
                    )
                else:
                    # It's a noise cluster
                    noise_count = len(
                        [i for i, label in enumerate(labels_str) if label == sub_label]
                    )
                    logger.info(f"    └─ {sub_label}: {noise_count} pts")

        logger.info(f"\nTotal new sub-clusters created: {total_sub_clusters}")

        # Calculate density improvements
        density_improvements = []
        for _parent_id, info in refinement_tree.items():
            original_density = info["original_density"]
            for sub_label in info["refined_into"]:
                if not sub_label.endswith(".noise"):
                    sub_stats = next(
                        (
                            c
                            for c in stats["clusters"]
                            if str(c["cluster_id"]) == sub_label
                        ),
                        None,
                    )
                    if sub_stats and original_density > 0:
                        improvement = sub_stats["density"] / original_density
                        density_improvements.append(improvement)

        if density_improvements:
            avg_improvement = sum(density_improvements) / len(density_improvements)
            logger.info(f"Average density improvement: {avg_improvement:.2f}x")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['overall']['n_samples']}")
    logger.info(f"Number of clusters: {stats['overall']['n_clusters']}")
    logger.info(
        f"Noise points: {stats['overall']['n_noise']} ({stats['overall']['noise_ratio'] * 100:.2f}%)"
    )
    if stats["overall"]["silhouette_score"] is not None:
        logger.info(f"Silhouette score: {stats['overall']['silhouette_score']:.4f}")

    logger.info("\nTop 10 clusters by size:")
    for i, cluster in enumerate(stats["clusters"][:10], 1):
        logger.info(
            f"  {i}. Cluster {cluster['cluster_id']}: "
            f"{cluster['size']} points ({cluster['percentage']:.2f}%), "
            f"compactness={cluster['compactness']:.4f}, "
            f"density={cluster['density']:.4f}"
        )

    # Save statistics to JSON (after adding visualization paths)
    save_cluster_statistics(stats, json_path)

    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis complete! Results saved to: {output_path}")
    logger.info(f"  - Statistics: {json_path}")
    logger.info(f"  - 2D plot: {output_path / 'clusters_2d.png'}")
    logger.info(f"  - 3D plot: {output_path / 'clusters_3d.png'}")
    logger.info(f"  - Sample images: {samples_dir}")
    logger.info("=" * 60)
