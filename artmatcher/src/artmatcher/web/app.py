import logging
import random
from pathlib import Path
from typing import cast

from flask import Flask, Response, render_template, request, send_file

from common import require_auth

from .session import LabelSession
from .source import ClassificationSource, MatchItem

logger = logging.getLogger(__name__)


def create_app(
    source: ClassificationSource,
    *,
    threshold: float,
    auth: bool,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=Path(__file__).parent.parent / "templates",
    )

    if auth:
        require_auth(app)

    order = list(range(source.total_images))
    random.shuffle(order)
    logger.info(f"Shuffled {len(order)} images for random traversal")

    session = LabelSession(threshold=threshold, order=order)

    def _context(item: MatchItem | None) -> dict:
        if item is None:
            return {
                "error": "No images found.",
                "current_index": 0,
                "total_images": source.total_images,
                "glashelder_path": "",
                "wikiart_path": "",
                "top_match_index": 0,
                "similarities": [],
                "is_match": False,
                "diff": 0.0,
                "threshold": session.threshold,
                "dataset_counts": session.dataset.counts(),
                # no item: the page renders an error, so the badge is unused
                "classifier_type": "",
                "features": {},
            }

        wikiart_idx = item.top_indices[0]
        return {
            "error": None,
            "current_index": item.glashelder_index,
            "total_images": source.total_images,
            "glashelder_path": item.glashelder_path,
            "wikiart_path": source.wikiart_path(wikiart_idx),
            "top_match_index": wikiart_idx,
            "similarities": item.similarities,
            "is_match": item.is_match,
            "diff": item.diff,
            "threshold": item.threshold,
            "dataset_counts": session.dataset.counts(),
            "classifier_type": item.classifier_type,
            "features": item.features,
        }

    def _render_content(item: MatchItem | None, want_match: bool) -> str:
        if item is None:
            return render_template(
                "content.html",
                error=f"No more {'matches' if want_match else 'negatives'} to show.",
                dataset_counts=session.dataset.counts(),
            )
        ctx = _context(item)
        return render_template("content.html", **ctx)

    def _find_next(start_queue_pos: int, want_match: bool) -> MatchItem | None:
        batch_size = source.batch_size
        queue_pos = start_queue_pos
        while queue_pos < len(session.order):
            chunk_positions = list(
                range(queue_pos, min(queue_pos + batch_size, len(session.order)))
            )
            chunk_indices = [session.order[p] for p in chunk_positions]
            items = source.classify_many(chunk_indices, session.threshold)
            for pos, item in zip(chunk_positions, items, strict=True):
                if item is not None and item.is_match == want_match:
                    session.queue_position = pos
                    session.current_item = item
                    return item
            queue_pos += batch_size
        return None

    def _label(is_correct: bool) -> str:
        item = cast("MatchItem | None", session.current_item)
        if item is not None:
            wikiart_idx = item.top_indices[0]
            entry = {
                "glashelder_index": item.glashelder_index,
                "glashelder_path": item.glashelder_path,
                "wikiart_index": wikiart_idx,
                "wikiart_path": source.wikiart_path(wikiart_idx),
                "diff": item.diff,
                "threshold": item.threshold,
                "similarities": item.similarities,
            }
            if is_correct:
                if item.is_match:
                    session.dataset.positives.append(entry)
                else:
                    session.dataset.negatives.append(entry)
            else:
                if item.is_match:
                    session.dataset.false_positives.append(entry)
                else:
                    session.dataset.false_negatives.append(entry)

        next_item = _find_next(session.queue_position + 1, True)
        if next_item is None:
            next_item = _find_next(session.queue_position + 1, False)
        return _render_content(next_item, True)

    @app.route("/")
    def index():
        item = _find_next(0, True)
        if item is None and session.order:
            item = source.classify(session.order[0], session.threshold)
            session.queue_position = 0
            session.current_item = item

        ctx = _context(item)
        ctx["error"] = None if item else "No images found."
        return render_template("labeler.html", **ctx)

    @app.route("/next-match", methods=["POST"])
    def next_match():
        item = _find_next(session.queue_position + 1, True)
        return _render_content(item, True)

    @app.route("/next-negative", methods=["POST"])
    def next_negative():
        item = _find_next(session.queue_position + 1, False)
        return _render_content(item, False)

    @app.route("/set-threshold", methods=["POST"])
    def set_threshold():
        new_threshold = request.form.get("threshold", type=float)
        if new_threshold is not None:
            session.threshold = new_threshold
        current = cast("MatchItem | None", session.current_item)
        if current:
            item = source.classify(current.glashelder_index, session.threshold)
            session.current_item = item
        else:
            item = None
        return _render_content(item, True)

    @app.route("/label-correct", methods=["POST"])
    def label_correct():
        return _label(True)

    @app.route("/label-incorrect", methods=["POST"])
    def label_incorrect():
        return _label(False)

    @app.route("/download-dataset")
    def download_dataset():
        return Response(
            session.dataset.to_json(),
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment;filename=labeling_dataset.json"
            },
        )

    @app.route("/glashelder-image/<int:index>")
    def serve_glashelder_image(index: int):
        path = source.glashelder_path(index)
        if path is None:
            return "Image not found", 404
        return send_file(path)

    @app.route("/wikiart-image/<int:index>")
    def serve_wikiart_image(index: int):
        path = source.wikiart_path(index)
        if not path:
            return "Image not found", 404
        return send_file(path)

    return app
