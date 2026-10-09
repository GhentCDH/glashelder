"""The validation viewer: browse the combined risk assessment results per image."""

import csv
import logging
from pathlib import Path

from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

logger = logging.getLogger(__name__)


def load_results(csv_path: Path) -> list[dict[str, str]]:
    """Load results from the combined CSV file, or nothing if it does not exist yet."""
    if not csv_path.is_file():
        logger.warning(f"No results at {csv_path}, run `glaswacht run` first")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    logger.info(f"Loaded {len(results)} results from {csv_path}")
    return results


def create_app(csv_path: Path, images_dir: Path) -> Flask:
    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

    results = load_results(csv_path)

    def render(index: int):
        return render_template(
            "validate.html",
            result=results[index],
            index=index,
            total=len(results),
        )

    @app.route("/")
    def index():
        """Show the first result."""
        if not results:
            return f"No results found in {csv_path}", 404
        return render(0)

    @app.route("/result/<int:index>")
    def show_result(index):
        """Show a specific result by index."""
        if index < 0 or index >= len(results):
            return "Result not found", 404
        return render(index)

    @app.route("/image/<path:filename>")
    def serve_image(filename):
        return send_from_directory(images_dir, filename)

    @app.route("/reload")
    def reload():
        """Reload the CSV file and redirect back."""
        results[:] = load_results(csv_path)

        referer = request.referrer
        if referer and "/result/" in referer:
            try:
                index = int(referer.split("/result/")[-1].split("?")[0])
                index = min(index, max(0, len(results) - 1))
                return redirect(url_for("show_result", index=index))
            except ValueError:
                pass

        return redirect(url_for("index"))

    return app
