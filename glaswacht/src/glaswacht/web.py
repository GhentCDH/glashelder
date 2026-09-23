import csv
import logging
import os
from pathlib import Path

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration — set RA_CSV_PATH / RA_IMAGES_DIR env vars to override.
# The default is relative to the working directory, matching where
# `write_combined_csv` puts it, so run-then-serve from one directory works.
CSV_PATH = Path(os.environ.get("RA_CSV_PATH", "combined_risk_assessment_results.csv"))
IMAGES_DIR = Path(
    os.environ.get(
        "RA_IMAGES_DIR",
        "/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop",
    )
)
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Global results list
results = []


def load_results():
    """Load results from the combined CSV file.

    Never raises: a missing CSV must not take the server down, since it is
    commonly started before the pipeline has written any results.
    """
    global results
    try:
        with open(CSV_PATH) as f:
            results = list(csv.DictReader(f))
    except OSError as exc:
        results = []
        logger.warning("Could not read %s: %s", CSV_PATH.resolve(), exc)
        return
    logger.info("Loaded %d results from %s", len(results), CSV_PATH.resolve())


def _no_results_page() -> tuple[str, int]:
    return (
        "<h1>No results loaded</h1>"
        f"<p>Looked for <code>{CSV_PATH.resolve()}</code>.</p>"
        "<p>Point the server at the right file with "
        "<code>glaswacht serve --csv-path &lt;file&gt;</code> "
        "(or the <code>RA_CSV_PATH</code> environment variable), then "
        '<a href="/reload">reload</a>.</p>',
        404,
    )


@app.route("/")
def index():
    """Show the first result."""
    if not results:
        load_results()
    if not results:
        return _no_results_page()

    return render_template(
        "validate.html",
        result=results[0],
        index=0,
        total=len(results),
    )


@app.route("/result/<int:index>")
def show_result(index):
    """Show a specific result by index."""
    if not results:
        load_results()
    if not results:
        return _no_results_page()
    if index < 0 or index >= len(results):
        return "Result not found", 404

    return render_template(
        "validate.html",
        result=results[index],
        index=index,
        total=len(results),
    )


@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve an image file."""
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return "Image not found", 404

    return send_from_directory(image_path.parent, image_path.name)


@app.route("/reload")
def reload():
    """Reload the CSV file and redirect back."""
    load_results()

    referer = request.referrer
    if referer and "/result/" in referer:
        try:
            index = int(referer.split("/result/")[-1])
            if index >= len(results):
                index = max(0, len(results) - 1)
            return redirect(url_for("show_result", index=index))
        except (ValueError, IndexError):
            pass

    return redirect(url_for("index"))


def main():
    """Start the Flask development server."""
    logger.info("Starting validation server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()
