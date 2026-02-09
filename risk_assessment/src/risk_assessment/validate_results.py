import csv
from pathlib import Path
from flask import Flask, render_template, send_from_directory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CSV_PATH = "/home/bas/Documents/Visual Code Repos/glashelder/risk_assessment/src/risk_assessment/lavaguard_risk_assessment_results.csv"
IMAGES_DIR = Path("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Load results once
results = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

logger.info(f"Loaded {len(results)} results from {CSV_PATH}")


@app.route("/")
def index():
    """Show the first result."""
    if not results:
        return "No results found", 404

    return render_template(
        "validate.html",
        result=results[0],
        index=0,
        total=len(results)
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
        total=len(results)
    )


@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve an image file."""
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return "Image not found", 404

    return send_from_directory(image_path.parent, image_path.name)


def main():
    """Start the Flask development server."""
    logger.info("Starting validation server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()
