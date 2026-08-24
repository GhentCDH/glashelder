"""CLI for matching images from the Glashelder dataset to images from scraped databases (like WikiArt or KIK-IRPA)"""

import glob
import json
import logging
from pathlib import Path
from typing import Literal

import click
import cv2
import numpy as np
from PIL import Image

from common import ASSETS_DIR
from embed import get_device

from .similarity_classifier import SimpleClassifier
from .stitching import (
    extract_features,
    resize_max,
    stitch_images,
)

WINDOW = "image-matching"

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
def artmatcher():
    """Match glass plate photos to known artworks using feature matching and ML."""
    pass


@artmatcher.command()
@click.argument("image", type=str, required=True)
def keypoints(image: str, output: str):
    """Display SIFT keypoints for an image."""
    img = cv2.imread(image)
    assert img is not None, f"Failed to load image: {image}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()  # type:ignore
    kp = sift.detect(gray, None)
    kp_visual = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    show(kp_visual)
    cv2.destroyAllWindows()


@artmatcher.command()
@click.argument("image_one", type=str, required=True)
@click.argument("image_two", type=str, required=True)
@click.option("--size", type=int, help="max size of the images")
def match(image_one, image_two, size: int | None):
    """Match features between two images and display the result."""
    image_one_path = image_one
    image_two_path = image_two
    image_one = cv2.imread(image_one_path)
    image_two = cv2.imread(image_two_path)
    assert image_one is not None, f"Failed to load image: {image_one_path}"
    assert image_two is not None, f"Failed to load image: {image_two_path}"
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
    src_pts = np.array(
        [kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    dst_pts = np.array(
        [kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32
    ).reshape(-1, 1, 2)

    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    s1 = np.array([[scale_1, 0, 0], [0, scale_1, 0], [0, 0, 1]], dtype=np.float64)
    s2_inv = np.array(
        [[1 / scale_2, 0, 0], [0, 1 / scale_2, 0], [0, 0, 1]], dtype=np.float64
    )
    h = s2_inv @ h @ s1

    h1, w1 = image_one_orig.shape[:2]
    h2, w2 = image_two_orig.shape[:2]
    corners_1 = np.array(
        [[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_1, h)

    all_corners = np.concatenate(
        [
            warped_corners,
            np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(
                -1, 1, 2
            ),
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


@artmatcher.command()
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


@artmatcher.command()
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
    from .recognition import (
        build_database,
    )

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


@artmatcher.command()
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
    from .recognition import (
        RecognitionDatabase,
        find_painting,
    )

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
        assert query_img is not None, f"Failed to load query image: {query_image}"
        assert painting_img is not None, (
            f"Failed to load painting: {best.painting_path}"
        )

        # Extract features again for visualization
        sift = cv2.SIFT_create()  # type: ignore[attr-defined]
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


@artmatcher.command()
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
    "--workers",
    "-w",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel workers for ORB/hash feature extraction (1=sequential)",
)
@click.option(
    "--no-image-features",
    is_flag=True,
    help="Skip ORB/hash feature extraction (faster, but less accurate)",
)
def train_classifier(
    dataset: str,
    output_model: str,
    n_estimators: int,
    cv_folds: int,
    workers: int,
    no_image_features: bool,
):
    """Train an ML classifier from a labeled dataset.

    DATASET is the JSON file exported from the labeling web UI.
    OUTPUT_MODEL is where to save the trained model (.pkl file).

    The dataset should contain labeled examples in the format exported
    by the 'serve' command's "Download Dataset" button.

    Use --workers to parallelize ORB/hash feature extraction across CPU cores.
    For a 96-core machine, try --workers 48 or higher.

    Example:
        image-matching train-classifier labeling_dataset.json model.pkl --workers 48
    """
    from .ml_classifier import (
        MLClassifier,
    )

    classifier = MLClassifier()

    try:
        classifier.train(
            dataset,
            n_estimators=n_estimators,
            cv_folds=cv_folds,
            extract_image_features=not no_image_features,
            workers=workers,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    classifier.save(output_model)


@artmatcher.command()
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
    """Label images by computing similarity scores against a reference database."""
    import cluster
    import embed

    device = get_device()
    data = embed.load_embeddings(embeddings)

    if data.get("model", "clip") != model:
        logger.error(
            "Embeddings should be from the same model as the model used for the embedder of glashelder images (the --model option)"
        )
        return

    embeddings = data["embeddings"].to(device)
    wikiart_paths: list[str] = data["images"]

    embedder = embed.get_embedder(model)
    embedder.load(device)

    image_paths = glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

    matcher = SimpleClassifier()

    for image_path in image_paths:
        image = Image.open(image_path)

        embedding = embedder.encode_pil_image(image)

        top_indices, top_values = cluster.perform_search(
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


@artmatcher.command()
@click.option(
    "--wikiart-embeddings",
    "-w",
    type=str,
    default=None,
    help="Path to the wikiart embeddings '.pkl' file (not needed with --results-json)",
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
    "--ml-model",
    type=click.Path(exists=True),
    default=None,
    help="Path to trained ML model (.pkl file). Required if --classifier=ml",
)
@click.option(
    "--search-workers",
    "-sw",
    type=int,
    default=8,
    show_default=True,
    help="Number of parallel workers for ORB/hash extraction during search (ML classifier only)",
)
@click.option(
    "--search-batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Number of images to check in parallel when searching for matches",
)
@click.option(
    "--results-json",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Path to pre-computed batch_classify JSON output. Skips on-the-fly computation.",
)
@click.option(
    "--glashelder-images",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Path of the directory where the glashelder images live",
)
@click.option(
    "--reference-images",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Path of the directory where the reference images live",
)
@click.option(
    "--auth", is_flag=True, help="Enable secret url parameter based authentication."
)
@click.option(
    "--debug", is_flag=True, help="Enable the Flask debugger and auto-reloader."
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
    ml_model: str | None,
    search_workers: int,
    search_batch_size: int,
    results_json: str | None,
    glashelder_images: Path | None,
    reference_images: Path | None,
    auth: bool,
    debug: bool,
):
    """Start a web interface for labeling artwork matches.

    Authentication is required. Set the GLASHELDER_SECRET environment variable,
    then access the server with /?secret=<your-secret> to get a session cookie.

    Use --results-json to load pre-computed results from batch_classify.
    The threshold can still be adjusted interactively in the UI.

    Or use --classifier=ml with --ml-model to compute on the fly.
    Use --search-workers to parallelize ORB/hash extraction during search.
    """
    if glashelder_images is None:
        logger.error(
            "!! You need to set the `--glashelder-images` argument to the directory where the glashelder images are on your system"
        )
        return
    else:
        glashelder_images = Path(glashelder_images)

    if reference_images is None:
        logger.error(
            "!! You need to set the `--reference-images` argument to the directory where the reference images are on your system"
        )
        return
    else:
        reference_images = Path(reference_images)

    from .web.app import create_app

    # If results_json is provided, use pre-computed results mode
    if results_json is not None:
        from .web.source import PrecomputedSource

        logger.info(f"Loading pre-computed results from {results_json}...")
        with open(results_json) as f:
            precomputed_data = json.load(f)

        results = precomputed_data["results"]
        logger.info(f"Loaded {len(results)} pre-computed results")

        source = PrecomputedSource(results, glashelder_images, reference_images)
    else:
        from .loading import load_classifier, load_matching_inputs
        from .web.source import LiveSource

        # `None` when ml_model isn't given
        ml_classifier_instance = load_classifier(ml_model, threshold)
        try:
            inputs = load_matching_inputs(
                wikiart_embeddings=wikiart_embeddings,
                glashelder_embeddings=glashelder_embeddings,
                model=model,
                image_dir=image_dir,
                wikiart_root=reference_images,
                glashelder_root=glashelder_images,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            return

        source = LiveSource(
            inputs,
            top_k=top_k,
            classifier=ml_classifier_instance,
            search_workers=search_workers,
            search_batch_size=search_batch_size,
        )

    app = create_app(source, threshold=threshold, auth=auth)

    logger.info(f"Starting Flask server on {host}:{port}...")
    logger.info(f"Open http://localhost:{port} in your browser")
    app.run(host=host, port=port, debug=debug)


@artmatcher.command()
@click.option(
    "--wikiart-embeddings",
    "-w",
    type=click.Path(exists=True),
    required=True,
    help="Path to wikiart embeddings pickle file",
)
@click.option(
    "--ml-model",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained ML classifier (.pkl file)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path (.json or .csv)",
)
@click.option(
    "--glashelder-embeddings",
    "-g",
    type=click.Path(exists=True),
    default=None,
    help="Pre-computed glashelder embeddings (generates if not provided)",
)
@click.option(
    "--image-dir",
    "-i",
    type=str,
    default=ASSETS_DIR,
    help="Path to glashelder images",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "csv", "both"]),
    default="json",
    show_default=True,
    help="Output format",
)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Parallel workers for ORB extraction",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    show_default=True,
    help="Top wikiart matches to consider",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Confidence threshold for is_match",
)
@click.option(
    "--min-confidence",
    type=float,
    default=None,
    help="Only output results above this confidence",
)
@click.option(
    "--cache-features",
    is_flag=True,
    help="Cache features to disk for reuse",
)
@click.option(
    "--cache-path",
    type=click.Path(),
    default="./feature_cache.pkl",
    show_default=True,
    help="Path for feature cache",
)
@click.option(
    "--cache-save-interval",
    type=int,
    default=100,
    show_default=True,
    help="Save cache to disk every N items (0 to disable periodic saves)",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Timeout in seconds per item for feature extraction",
)
@click.option(
    "--model",
    default="clip",
    type=click.Choice(["clip", "dinov2", "qwen3vl"]),
    help="Embedding model to use if generating embeddings",
)
def batch_classify(
    wikiart_embeddings: str,
    ml_model: str,
    output: str,
    glashelder_embeddings: str | None,
    image_dir: str,
    output_format: Literal["json", "csv", "both"],
    workers: int,
    top_k: int,
    threshold: float,
    min_confidence: float | None,
    cache_features: bool,
    cache_path: str,
    cache_save_interval: int,
    timeout: float,
    model: Literal["clip", "dinov2", "qwen3vl"],
):
    """Batch classify all glashelder images for artwork matches.

    This command processes all glashelder images through the artwork matcher,
    computing embedding similarities and running ML classification with
    parallel ORB/hash feature extraction.

    Example:
        image-matching batch-classify \\
            -w wikiart.pkl \\
            --ml-model model.pkl \\
            -o results.json \\
            --workers 32
    """
    from .batch import (
        FeatureCache,
        write_csv_output,
        write_json_output,
    )
    from .batch import (
        batch_classify as do_batch_classify,
    )
    from .loading import load_classifier, load_matching_inputs

    classifier = load_classifier(ml_model, threshold)
    assert classifier is not None
    # NOTE: batch_classify passes no wikiart root (unlike `serve`, which passes
    # --reference-images). Preserved as-is; changing it is out of scope.
    try:
        inputs = load_matching_inputs(
            wikiart_embeddings=wikiart_embeddings,
            glashelder_embeddings=glashelder_embeddings,
            model=model,
            image_dir=image_dir,
            wikiart_root=None,
            glashelder_root=image_dir,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Setup feature cache
    cache = FeatureCache(cache_path) if cache_features else None

    # Run batch classification
    result = do_batch_classify(
        glashelder_emb=inputs.glashelder_emb,
        glashelder_paths=inputs.glashelder_paths,
        wikiart_emb=inputs.wikiart_emb,
        wikiart_paths=inputs.wikiart_paths,
        classifier=classifier,
        top_k=top_k,
        threshold=threshold,
        workers=workers,
        min_confidence=min_confidence,
        cache=cache,
        timeout_per_item=timeout,
        cache_save_interval=cache_save_interval,
    )

    # Write output
    output_path = Path(output)
    if output_format == "json":
        write_json_output(result, output_path)
    elif output_format == "csv":
        csv_path = output_path.with_suffix(".csv")
        write_csv_output(result, csv_path)
    else:  # both
        write_json_output(result, output_path.with_suffix(".json"))
        write_csv_output(result, output_path.with_suffix(".csv"))

    # Save cache if enabled
    if cache:
        cache.save()

    logger.info("Batch classification complete!")


@artmatcher.command()
@click.option(
    "--input",
    "-i",
    type=str,
    help="The `json` file containing the results from batch-classify",
)
@click.option(
    "--output-dir",
    "-o",
    type=str,
    help="The output directory where plots' pngs will be stored",
)
def plot_results(input, output_dir):
    """Generate analysis plots from batch classification results."""
    from .plot import (
        extract_arrays,
        load_data,
        plot_confidence_distribution,
        plot_feature_correlations,
        plot_hash_orb,
        plot_statistical_analysis,
        setup_style,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input} ...")
    data = load_data(input)
    results = data["results"]
    meta = data.get("metadata", {})
    summary = data.get("summary", {})
    logger.info(f"  {len(results)} results loaded")

    d = extract_arrays(results)
    d["is_match"] = d["is_match"].astype(bool)

    setup_style()

    logger.info("Generating plots ...")
    plot_confidence_distribution(d, meta, summary, output_dir)
    plot_feature_correlations(d, output_dir)
    plot_hash_orb(d, output_dir)
    plot_statistical_analysis(d, output_dir)
    logger.info("Done!")
