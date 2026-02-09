"""CLI for image matching and stitching."""

import glob
import json
import logging
import random
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from PIL import Image

from common import ASSETS_DIR
from common.clipmodel import get_device
from image_matching.ml_classifier import MLClassifier
from image_matching.recognition import (
    RecognitionDatabase,
    build_database,
    find_painting,
)
from image_matching.similarity_classifier import SimpleClassifier
from image_matching.stitching import (
    extract_features,
    resize_max,
    stitch_images,
)

WINDOW = "image-matching"
PACKAGE_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


def show(
    image,
    window: str = WINDOW,
) -> int | None:
    """Show an image and wait for 'q' to close."""
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error:
        pass

    image = np.copy(image)
    cv2.imshow(window, image)

    while cv2.waitKey(1) & 0xFF != ord("q"):
        pass


@click.group()
def image_matching():
    pass


@image_matching.command()
@click.argument("image", type=str, required=True)
def keypoints(image: str, output: str):
    """Display SIFT keypoints for an image."""
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()  # type:ignore
    kp = sift.detect(gray, None)
    kp_visual = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    show(kp_visual)
    cv2.destroyAllWindows()


@image_matching.command()
@click.argument("image_one", type=str, required=True)
@click.argument("image_two", type=str, required=True)
@click.option("--size", type=int, help="max size of the images")
def match(image_one, image_two, size: int | None):
    """Match features between two images and display the result."""
    image_one = cv2.imread(image_one)
    image_two = cv2.imread(image_two)
    image_one_orig = np.copy(image_one)
    image_two_orig = np.copy(image_two)

    scale_1 = 1.0
    scale_2 = 1.0
    if size is not None:
        _, w1_orig = image_one.shape[:2]
        _, w2_orig = image_two.shape[:2]
        image_one = resize_max(image_one, size)
        image_two = resize_max(image_two, size)
        scale_1 = image_one.shape[1] / w1_orig
        scale_2 = image_two.shape[1] / w2_orig

    sift = cv2.SIFT_create()  # type:ignore
    bf = cv2.BFMatcher()

    kp1, des1 = extract_features(image_one, sift)
    kp2, des2 = extract_features(image_two, sift)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    img3 = cv2.drawMatches(  # type:ignore
        image_one,
        kp1,
        image_two,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    show(img3)

    # Compute and display stitched result
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type:ignore
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # type:ignore

    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    s1 = np.array([[scale_1, 0, 0], [0, scale_1, 0], [0, 0, 1]], dtype=np.float64)
    s2_inv = np.array(
        [[1 / scale_2, 0, 0], [0, 1 / scale_2, 0], [0, 0, 1]], dtype=np.float64
    )
    h = s2_inv @ h @ s1

    h1, w1 = image_one_orig.shape[:2]
    h2, w2 = image_two_orig.shape[:2]
    corners_1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)  # type:ignore
    warped_corners = cv2.perspectiveTransform(corners_1, h)

    all_corners = np.concatenate(
        [
            warped_corners,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2),  # type:ignore
        ]
    )

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())  # ty:ignore[not-iterable]
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())  # ty:ignore[not-iterable]

    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64
    )

    canvas_size = (x_max - x_min, y_max - y_min)
    stitched = cv2.warpPerspective(image_one_orig, translation @ h, canvas_size)
    stitched[-y_min : -y_min + h2, -x_min : -x_min + w2] = image_two_orig

    show(stitched)
    cv2.destroyAllWindows()


@image_matching.command()
@click.argument("images", type=str, nargs=-1, required=True)
@click.option("--size", type=int, help="max size of the images for feature matching")
@click.option(
    "--max-canvas",
    type=int,
    default=8000,
    show_default=True,
    help="max canvas dimension (scales down if exceeded to limit memory)",
)
@click.option(
    "--max-stretch",
    type=float,
    default=3.0,
    show_default=True,
    help="max allowed stretch ratio for warped images (skip images that exceed this)",
)
def stitch(
    images: tuple[str, ...],
    size: int | None,
    max_canvas: int | None,
    max_stretch: float,
):
    """Stitch multiple images together into a panorama.

    The order of image arguments doesn't matter - the algorithm will
    find the best arrangement based on feature matching.

    Images that would be warped too much (stretch ratio > max-stretch) are
    automatically skipped to avoid distorted results.
    """
    if len(images) < 2:
        click.echo("Error: Need at least 2 images to stitch", err=True)
        return

    def on_progress(msg: str) -> None:
        # Add image names to messages where relevant
        click.echo(msg)

    try:
        result = stitch_images(
            list(images),
            feature_size=size,
            max_canvas=max_canvas,
            max_stretch=max_stretch,
            on_progress=on_progress,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Report results
    click.echo(f"Base image: {Path(images[result.base_index]).name}")

    if result.skipped_stretch:
        names = [Path(images[i]).name for i in result.skipped_stretch]
        click.echo(f"Skipped due to excessive warping: {names}")

    if result.skipped_no_connection:
        names = [Path(images[i]).name for i in result.skipped_no_connection]
        click.echo(f"Could not connect: {names}", err=True)

    click.echo("Done! Press 'q' to close the window.")
    show(result.image)
    cv2.destroyAllWindows()


@image_matching.command()
@click.argument("painting_dir", type=click.Path(exists=True), required=True)
@click.argument("output_db", type=click.Path(), required=True)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    show_default=True,
    help="number of parallel workers for feature extraction (1=sequential)",
)
@click.option(
    "--max-size",
    type=int,
    default=None,
    help="maximum dimension for resizing images before feature extraction (faster, uses less memory)",
)
@click.option(
    "--max-features",
    type=int,
    default=None,
    help="maximum keypoints per image (e.g., 1000). Drastically reduces database size. Recommended: 500-2000",
)
def build_db(
    painting_dir: str,
    output_db: str,
    workers: int,
    max_size: int | None,
    max_features: int | None,
):
    """Build a recognition database from a directory of known paintings.

    PAINTING_DIR should contain images of known paintings (JPG/PNG).
    OUTPUT_DB is the path where the database will be saved (.pkl file).

    Use --workers to parallelize feature extraction across multiple CPU cores.
    Use --max-features to limit keypoints per image (CRITICAL for controlling database size!)
    Use --max-size to resize images before feature extraction (faster, lower memory).

    Examples:
        # Recommended: Limit features to keep database small (~0.5MB per image)
        image-matching build-db ./paintings ./db.pkl --workers 4 --max-features 1000

        # Smaller database with resizing (~0.2-0.3MB per image)
        image-matching build-db ./paintings ./db.pkl --workers 4 --max-size 2000 --max-features 500

        # Default (unlimited features, can create 50GB+ databases!)
        image-matching build-db ./paintings ./db.pkl --workers 4
    """
    painting_dir_path = Path(painting_dir)

    # Find all image files
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    painting_paths = []
    for ext in extensions:
        painting_paths.extend(painting_dir_path.glob(ext))

    if not painting_paths:
        click.echo(f"Error: No images found in {painting_dir}", err=True)
        return

    click.echo(f"Found {len(painting_paths)} images in {painting_dir}")

    def on_progress(msg: str) -> None:
        click.echo(msg)

    # Build database with parallel workers
    db = build_database(
        [str(p) for p in painting_paths],
        workers=workers,
        max_size=max_size,
        max_features=max_features,
        on_progress=on_progress,
    )

    # Save to disk
    db.save(output_db)
    click.echo(f"Database saved to {output_db}")


@image_matching.command()
@click.argument("query_image", type=click.Path(exists=True), required=True)
@click.argument("database", type=click.Path(exists=True), required=True)
@click.option(
    "--min-matches",
    type=int,
    default=10,
    show_default=True,
    help="minimum number of feature matches required",
)
@click.option(
    "--min-inliers",
    type=int,
    default=4,
    show_default=True,
    help="minimum number of inliers after geometric verification",
)
@click.option(
    "--max-results",
    type=int,
    default=5,
    show_default=True,
    help="maximum number of results to show",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="skip geometric verification (faster but less accurate)",
)
@click.option(
    "--show-matches",
    is_flag=True,
    help="visualize matches for the best result",
)
def identify(
    query_image: str,
    database: str,
    min_matches: int,
    min_inliers: int,
    max_results: int,
    no_verify: bool,
    show_matches: bool,
):
    """Identify which known painting appears in a query image.

    QUERY_IMAGE is the photo you want to identify (e.g., a photo of a painting).
    DATABASE is the pre-built database file (created with build-db command).

    Example:
        image-matching identify photo.jpg paintings.db.pkl --show-matches
    """
    # Load database
    click.echo(f"Loading database from {database}...")
    db = RecognitionDatabase.load(database)
    click.echo(f"Database contains {len(db.features)} paintings")

    # Find matches
    click.echo("Searching for matches in query image...")
    results = find_painting(
        query_image,
        db,
        min_matches=min_matches,
        min_inliers=min_inliers,
        max_results=max_results,
        verify_geometry=not no_verify,
    )

    if not results:
        click.echo("No matches found.", err=True)
        return

    # Display results
    click.echo(f"\nFound {len(results)} match(es):\n")
    for i, result in enumerate(results, 1):
        click.echo(f"{i}. {result.painting_name}")
        click.echo(f"   Matches: {result.num_matches}")
        click.echo(f"   Inliers: {result.num_inliers}")
        click.echo(f"   Confidence: {result.confidence:.2%}")
        click.echo()

    # Visualize best match if requested
    if show_matches and results:
        best = results[0]
        click.echo(f"Visualizing matches for: {best.painting_name}")
        click.echo("Press 'q' to close the window.")

        # Load images
        query_img = cv2.imread(query_image)
        painting_img = cv2.imread(best.painting_path)

        # Extract features again for visualization
        sift = cv2.SIFT_create()  # type:ignore
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

        query_kp, query_des = sift.detectAndCompute(query_gray, None)
        painting_kp, painting_des = db.features[best.painting_path]

        # Re-match for visualization
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_des, painting_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Draw matches
        match_img = cv2.drawMatches(  # type:ignore
            query_img,
            query_kp,
            painting_img,
            painting_kp,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        show(match_img, window="Matches")
        cv2.destroyAllWindows()


@image_matching.command()
@click.argument("dataset", type=click.Path(exists=True), required=True)
@click.argument("output_model", type=click.Path(), required=True)
@click.option(
    "--n-estimators",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees in the Random Forest",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation folds",
)
@click.option(
    "--no-sift",
    is_flag=True,
    help="Skip SIFT feature extraction (faster, but less accurate)",
)
def train_classifier(
    dataset: str,
    output_model: str,
    n_estimators: int,
    cv_folds: int,
    no_sift: bool,
):
    """Train an ML classifier from a labeled dataset.

    DATASET is the JSON file exported from the labeling web UI.
    OUTPUT_MODEL is where to save the trained model (.pkl file).

    The dataset should contain labeled examples in the format exported
    by the 'serve' command's "Download Dataset" button.

    Example:
        image-matching train-classifier labeling_dataset.json model.pkl
    """
    click.echo(f"Loading dataset from {dataset}...")

    classifier = MLClassifier()

    try:
        metrics = classifier.train(
            dataset,
            n_estimators=n_estimators,
            cv_folds=cv_folds,
            extract_sift=not no_sift,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    click.echo("\nTraining Results:")
    click.echo(
        f"  Samples: {metrics.n_samples} ({metrics.n_positives} positives, {metrics.n_negatives} negatives)"
    )
    click.echo(f"  CV Accuracy: {metrics.cv_accuracy:.3f} (+/- {metrics.cv_std:.3f})")
    click.echo("\nTop 5 Features by Importance:")
    for i, (name, imp) in enumerate(list(metrics.feature_importance.items())[:5]):
        click.echo(f"  {i + 1}. {name}: {imp:.4f}")

    classifier.save(output_model)
    click.echo(f"\nModel saved to {output_model}")


@image_matching.command()
@click.option(
    "--embeddings", "-e", type=str, help="Path to the wikiart embeddings '.pkl' file"
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use if generating embeddings",
)
@click.option(
    "--image-dir",
    "-i",
    type=str,
    default=ASSETS_DIR,
    help="Path to the Glashelder images",
)
def label(embeddings: str, model: Literal["clip", "dinov2", "qwen3vl"], image_dir: str):
    import clipcluster

    """Create labels for images' similarity score plots"""

    device = get_device()
    data = clipcluster.load_embeddings(embeddings)

    if data.get("model", "clip") != model:
        logger.error(
            "Embeddings should be from the same model as the model used for the embedder of glashelder images (the --model option)"
        )
        return

    embeddings = data["embeddings"].to(device)
    wikiart_paths: list[str] = data["images"]

    embedder = clipcluster.get_embedder(model)
    embedder.load(device)

    image_paths = glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

    matcher = SimpleClassifier()

    for image_path in image_paths:
        image = Image.open(image_path)

        embedding = embedder.encode_pil_image(image)

        top_indices, top_values = clipcluster.perform_search(
            10, embedding, embeddings, image_paths
        )

        indexes = []
        similarities = []
        for idx, score in zip(
            top_indices.cpu().numpy(), top_values.cpu().numpy(), strict=False
        ):
            indexes.append(idx)
            similarities.append(score)

        is_known_painting = matcher.classify(similarities)

        logger.info(f"Is known painting: {is_known_painting}")

        if is_known_painting:
            logger.info(
                f"MATCHMATCHMATCH: \n{image_path} \nmatches \n{wikiart_paths[indexes[0]]}"
            )


@image_matching.command()
@click.option(
    "--wikiart-embeddings",
    "-w",
    type=str,
    required=True,
    help="Path to the wikiart embeddings '.pkl' file",
)
@click.option(
    "--glashelder-embeddings",
    "-g",
    type=str,
    default=None,
    help="Path to pre-calculated glashelder embeddings '.pkl' file (optional)",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use",
)
@click.option(
    "--image-dir",
    "-i",
    type=str,
    default=ASSETS_DIR,
    help="Path to the Glashelder images",
)
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--top-k", default=10, help="Number of top matches to consider")
@click.option(
    "--threshold", default=0.2, help="Diff threshold for match classification"
)
@click.option(
    "--classifier",
    type=click.Choice(["simple", "ml"]),
    default="simple",
    help="Classifier to use: 'simple' (polynomial fit) or 'ml' (Random Forest)",
)
@click.option(
    "--ml-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to trained ML model (.pkl file). Required if --classifier=ml",
)
def serve(
    wikiart_embeddings: str,
    glashelder_embeddings: str | None,
    model: Literal["clip", "dinov2", "qwen3vl"],
    image_dir: str,
    host: str,
    port: int,
    top_k: int,
    threshold: float,
    classifier: Literal["simple", "ml"],
    ml_model: str | None,
):
    """Start a web interface for labeling artwork matches.

    Use --classifier=ml with --ml-model to use a trained ML classifier.
    """
    from flask import Flask, render_template, send_file

    import clipcluster

    # Validate ML classifier options
    if classifier == "ml" and ml_model is None:
        click.echo("Error: --ml-model is required when using --classifier=ml", err=True)
        return

    app = Flask(
        __name__,
        template_folder=PACKAGE_ROOT / "templates",
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load ML classifier if specified
    ml_classifier_instance: MLClassifier | None = None
    if classifier == "ml" and ml_model:
        logger.info(f"Loading ML classifier from {ml_model}...")
        ml_classifier_instance = MLClassifier.load(ml_model, threshold=threshold)
        logger.info("ML classifier loaded")

    # Load wikiart embeddings
    logger.info(f"Loading wikiart embeddings from {wikiart_embeddings}...")
    wikiart_data = clipcluster.load_embeddings(wikiart_embeddings)
    wikiart_emb = wikiart_data["embeddings"].to(device)
    wikiart_paths: list[str] = wikiart_data["images"]
    embedding_model_name = wikiart_data.get("model", "clip")
    logger.info(f"Loaded {len(wikiart_paths)} wikiart images")
    logger.info(f"Embedding model: {embedding_model_name}")

    if embedding_model_name != model:
        logger.warning(
            f"Wikiart embeddings use model '{embedding_model_name}' but --model is '{model}'. "
            f"Using '{embedding_model_name}' for consistency."
        )
        model = embedding_model_name  # type: ignore

    # Load or generate glashelder embeddings
    if glashelder_embeddings is not None:
        logger.info(f"Loading glashelder embeddings from {glashelder_embeddings}...")
        glashelder_data = clipcluster.load_embeddings(glashelder_embeddings)
        glashelder_emb = glashelder_data["embeddings"].to(device)
        glashelder_paths: list[str] = glashelder_data["images"]
        logger.info(f"Loaded {len(glashelder_paths)} glashelder images")
    else:
        logger.info(f"Loading embedder model: {model}")
        embedder = clipcluster.get_embedder(model)
        embedder.load(device)

        glashelder_paths = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))
        logger.info(f"Found {len(glashelder_paths)} glashelder images")

        logger.info("Generating glashelder embeddings (this may take a while)...")
        glashelder_emb, glashelder_paths = embedder.encode_images(
            glashelder_paths, batch_size=64
        )
        glashelder_emb = glashelder_emb.to(device)
        logger.info(f"Generated embeddings for {len(glashelder_paths)} images")

    # Create shuffled order of indices
    shuffled_indices = list(range(len(glashelder_paths)))
    random.shuffle(shuffled_indices)
    logger.info(f"Shuffled {len(shuffled_indices)} images for random traversal")

    # State for tracking current position, threshold, and labeled dataset
    state: dict = {
        "queue_position": 0,  # Position in shuffled_indices
        "current_item": None,  # Current classified item
        "threshold": threshold,
        "dataset": {
            "positives": [],  # True matches (classifier correct when predicting match)
            "negatives": [],  # True negatives (classifier correct when predicting no match)
            "false_positives": [],  # Classifier said match but was wrong
            "false_negatives": [],  # Classifier said no match but was wrong
        },
    }

    def get_simple_classifier() -> SimpleClassifier:
        return SimpleClassifier(threshold=state["threshold"])

    def classify_image(image_index: int) -> dict | None:
        """Classify a single image by its index in glashelder_paths."""
        if image_index < 0 or image_index >= len(glashelder_paths):
            return None

        emb = glashelder_emb[image_index]
        top_indices, top_values = clipcluster.perform_search(
            top_k, emb.unsqueeze(0), wikiart_emb, wikiart_paths
        )
        similarities = top_values.cpu().numpy().tolist()
        top_indices_list = top_indices.cpu().numpy().tolist()

        glashelder_path = glashelder_paths[image_index]
        wikiart_path = wikiart_paths[top_indices_list[0]] if top_indices_list else None

        # Use ML classifier if available, otherwise simple classifier
        if ml_classifier_instance is not None:
            # Update threshold if changed in state
            ml_classifier_instance.threshold = state["threshold"]
            result = ml_classifier_instance.classify(
                similarities, glashelder_path, wikiart_path
            )
            return {
                "glashelder_index": image_index,
                "glashelder_path": glashelder_path,
                "top_indices": top_indices_list,
                "similarities": similarities,
                "is_match": result.is_match,
                "diff": result.confidence,  # Use confidence as "diff" for display
                "threshold": result.threshold,
                "classifier_type": "ml",
                "features": result.features,
            }
        else:
            simple_clf = get_simple_classifier()
            result = simple_clf.classify(similarities)
            return {
                "glashelder_index": image_index,
                "glashelder_path": glashelder_path,
                "top_indices": top_indices_list,
                "similarities": similarities,
                "is_match": result.is_match,
                "diff": result.diff,
                "threshold": result.threshold,
                "classifier_type": "simple",
            }

    def find_next_in_queue(start_queue_pos: int, want_match: bool) -> dict | None:
        """Find next image in shuffled queue that matches/doesn't match."""
        for queue_pos in range(start_queue_pos, len(shuffled_indices)):
            image_index = shuffled_indices[queue_pos]
            item = classify_image(image_index)
            if item is None:
                continue
            if item["is_match"] == want_match:
                state["queue_position"] = queue_pos
                state["current_item"] = item
                return item
        return None

    def get_dataset_counts() -> dict:
        """Get counts for each category in the dataset."""
        return {
            "positives": len(state["dataset"]["positives"]),
            "negatives": len(state["dataset"]["negatives"]),
            "false_positives": len(state["dataset"]["false_positives"]),
            "false_negatives": len(state["dataset"]["false_negatives"]),
            "total": sum(len(v) for v in state["dataset"].values()),
        }

    def render_content(item: dict | None, want_match: bool) -> str:
        if item is None:
            return render_template(
                "content.html",
                error=f"No more {'matches' if want_match else 'negatives'} to show.",
                dataset_counts=get_dataset_counts(),
            )

        return render_template(
            "content.html",
            error=None,
            current_index=item["glashelder_index"],
            total_images=len(glashelder_paths),
            glashelder_path=item["glashelder_path"],
            wikiart_path=wikiart_paths[item["top_indices"][0]]
            if item["is_match"]
            else None,
            top_match_index=item["top_indices"][0],
            similarities=item["similarities"],
            is_match=item["is_match"],
            diff=item["diff"],
            threshold=item["threshold"],
            dataset_counts=get_dataset_counts(),
        )

    @app.route("/")
    def index():
        # Find first match or first image in shuffled order
        item = find_next_in_queue(0, True)
        if item is None and shuffled_indices:
            item = classify_image(shuffled_indices[0])
            state["queue_position"] = 0
            state["current_item"] = item

        return render_template(
            "labeler.html",
            error=None if item else "No images found.",
            current_index=item["glashelder_index"] if item else 0,
            total_images=len(glashelder_paths),
            glashelder_path=item["glashelder_path"] if item else "",
            wikiart_path=wikiart_paths[item["top_indices"][0]]
            if item and item["is_match"]
            else None,
            top_match_index=item["top_indices"][0] if item else 0,
            similarities=item["similarities"] if item else [],
            is_match=item["is_match"] if item else False,
            diff=item["diff"] if item else 0.0,
            threshold=item["threshold"] if item else state["threshold"],
            dataset_counts=get_dataset_counts(),
        )

    @app.route("/next-match", methods=["POST"])
    def next_match():
        item = find_next_in_queue(state["queue_position"] + 1, True)
        return render_content(item, True)

    @app.route("/next-negative", methods=["POST"])
    def next_negative():
        item = find_next_in_queue(state["queue_position"] + 1, False)
        return render_content(item, False)

    @app.route("/set-threshold", methods=["POST"])
    def set_threshold():
        from flask import request

        new_threshold = request.form.get("threshold", type=float)
        if new_threshold is not None:
            state["threshold"] = new_threshold
        # Re-classify current image with new threshold
        if state["current_item"]:
            item = classify_image(state["current_item"]["glashelder_index"])
            state["current_item"] = item
        else:
            item = None
        return render_content(item, True)

    @app.route("/label-correct", methods=["POST"])
    def label_correct():
        """Mark current classification as correct and move to next."""
        item = state["current_item"]
        if item:
            entry = {
                "glashelder_index": item["glashelder_index"],
                "glashelder_path": item["glashelder_path"],
                "wikiart_index": item["top_indices"][0],
                "wikiart_path": wikiart_paths[item["top_indices"][0]],
                "diff": item["diff"],
                "threshold": item["threshold"],
                "similarities": item["similarities"],
            }
            if item["is_match"]:
                state["dataset"]["positives"].append(entry)
            else:
                state["dataset"]["negatives"].append(entry)

        # Move to next (prefer matches)
        next_item = find_next_in_queue(state["queue_position"] + 1, True)
        if next_item is None:
            next_item = find_next_in_queue(state["queue_position"] + 1, False)
        return render_content(next_item, True)

    @app.route("/label-incorrect", methods=["POST"])
    def label_incorrect():
        """Mark current classification as incorrect and move to next."""
        item = state["current_item"]
        if item:
            entry = {
                "glashelder_index": item["glashelder_index"],
                "glashelder_path": item["glashelder_path"],
                "wikiart_index": item["top_indices"][0],
                "wikiart_path": wikiart_paths[item["top_indices"][0]],
                "diff": item["diff"],
                "threshold": item["threshold"],
                "similarities": item["similarities"],
            }
            if item["is_match"]:
                state["dataset"]["false_positives"].append(entry)
            else:
                state["dataset"]["false_negatives"].append(entry)

        # Move to next (prefer matches)
        next_item = find_next_in_queue(state["queue_position"] + 1, True)
        if next_item is None:
            next_item = find_next_in_queue(state["queue_position"] + 1, False)
        return render_content(next_item, True)

    @app.route("/download-dataset")
    def download_dataset():
        from flask import Response

        dataset_json = json.dumps(state["dataset"], indent=2)
        return Response(
            dataset_json,
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment;filename=labeling_dataset.json"
            },
        )

    @app.route("/glashelder-image/<int:index>")
    def serve_glashelder_image(index: int):
        if index < 0 or index >= len(glashelder_paths):
            return "Image not found", 404
        return send_file(glashelder_paths[index])

    @app.route("/wikiart-image/<int:index>")
    def serve_wikiart_image(index: int):
        if index < 0 or index >= len(wikiart_paths):
            return "Image not found", 404
        return send_file(wikiart_paths[index])

    logger.info(f"Starting Flask server on {host}:{port}...")
    logger.info(f"Open http://localhost:{port} in your browser")
    app.run(host=host, port=port, debug=True)
