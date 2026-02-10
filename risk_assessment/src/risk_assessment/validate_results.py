import csv
from pathlib import Path
from flask import Flask, render_template, send_from_directory, redirect, url_for, request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CSV_PATH = "/home/bas/Documents/Visual Code Repos/glashelder/lavaguard_risk_assessment_results.csv"
CSV_PATH_MODEL2 = "/home/bas/Documents/Visual Code Repos/glashelder/qwen3_risk_assessment_results.csv"
CSV_PATH_MODEL3 = "/home/bas/Documents/Visual Code Repos/glashelder/pixtral_risk_assessment_results.csv"
IMAGES_DIR = Path("/mnt/UGent_Share/glashelder/Experimentele Dataset/contro-set-crop")
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Global results lists
results_model1 = []
results_model2 = []
results_model3 = []


def load_results():
    """Load results from CSV files."""
    global results_model1, results_model2, results_model3

    # Load model 1 results
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        results_model1 = list(reader)
    logger.info(f"Loaded {len(results_model1)} results from {CSV_PATH}")

    # Load model 2 results if path is configured
    if CSV_PATH_MODEL2 and Path(CSV_PATH_MODEL2).exists():
        with open(CSV_PATH_MODEL2, 'r') as f:
            reader = csv.DictReader(f)
            results_model2 = list(reader)
        logger.info(f"Loaded {len(results_model2)} results from {CSV_PATH_MODEL2}")
    else:
        results_model2 = []
        logger.info("Model 2 results not configured or file not found")

    # Load model 3 results if path is configured
    if CSV_PATH_MODEL3 and Path(CSV_PATH_MODEL3).exists():
        with open(CSV_PATH_MODEL3, 'r') as f:
            reader = csv.DictReader(f)
            results_model3 = list(reader)
        logger.info(f"Loaded {len(results_model3)} results from {CSV_PATH_MODEL3}")
    else:
        results_model3 = []
        logger.info("Model 3 results not configured or file not found")


# Load results at startup
load_results()


def get_model2_result(image_name):
    """Get the corresponding result from model 2 for the given image."""
    if not results_model2:
        return None

    for result in results_model2:
        if result.get('image') == image_name:
            return result
    return None


def get_model3_result(image_name):
    """Get the corresponding result from model 3 for the given image."""
    if not results_model3:
        return None

    for result in results_model3:
        if result.get('image') == image_name:
            return result
    return None


@app.route("/")
def index():
    """Show the first result."""
    if not results_model1:
        return "No results found", 404

    result_model2 = get_model2_result(results_model1[0].get('image'))
    result_model3 = get_model3_result(results_model1[0].get('image'))

    return render_template(
        "validate.html",
        result_model1=results_model1[0],
        result_model2=result_model2,
        result_model3=result_model3,
        index=0,
        total=len(results_model1),
        has_model2=bool(results_model2),
        has_model3=bool(results_model3)
    )


@app.route("/result/<int:index>")
def show_result(index):
    """Show a specific result by index."""
    if index < 0 or index >= len(results_model1):
        return "Result not found", 404

    result_model2 = get_model2_result(results_model1[index].get('image'))
    result_model3 = get_model3_result(results_model1[index].get('image'))

    return render_template(
        "validate.html",
        result_model1=results_model1[index],
        result_model2=result_model2,
        result_model3=result_model3,
        index=index,
        total=len(results_model1),
        has_model2=bool(results_model2),
        has_model3=bool(results_model3)
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

    # Get the referring page index, or default to 0
    referer = request.referrer
    if referer and '/result/' in referer:
        try:
            index = int(referer.split('/result/')[-1])
            # Make sure index is still valid after reload
            if index >= len(results):
                index = max(0, len(results) - 1)
            return redirect(url_for('show_result', index=index))
        except (ValueError, IndexError):
            pass

    return redirect(url_for('index'))


def main():
    """Start the Flask development server."""
    logger.info("Starting validation server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()
