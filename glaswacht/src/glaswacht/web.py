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

# Configuration — set RA_CSV_PATH / RA_IMAGES_DIR env vars to override
CSV_PATH = Path(
    os.environ.get(
        "RA_CSV_PATH", Path(__file__).parent / "combined_risk_assessment_results.csv"
    )
)
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
    """Load results from the combined CSV file."""
    global results
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        results = list(reader)
    logger.info(f"Loaded {len(results)} results from {CSV_PATH}")


# Load results at startup
load_results()


@app.route("/")
def index():
    """Show the first result."""
    if not results:
        return "No results found", 404

    return render_template(
        "validate.html",
        result=results[0],
        index=0,
        total=len(results),
    )


@app.route("/result/<int:index>")
def show_result(index):
    """Show a specific result by index."""
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
