"""CLI for image matching and stitching."""

from pathlib import Path

import click
import cv2
import numpy as np

from image_matching.recognition import (
    RecognitionDatabase,
    build_database,
    find_painting,
)
from image_matching.stitching import (
    extract_features,
    resize_max,
    stitch_images,
)

WINDOW = "image-matching"


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

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

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
