import json
import logging
import random
from pathlib import Path
from typing import cast

from flask import Flask, Response, render_template, request, send_file

from common import require_auth

from .session import LabelSession
from .source import ClassificationSource, MatchItem
from .store import (
    SessionStore,
    dataset_threshold,
    is_valid_session_id,
    new_session_id,
    normalize_dataset,
)

logger = logging.getLogger(__name__)


def create_app(
    source: ClassificationSource,
    *,
    threshold: float,
    auth: bool,
    session_dir: str | Path = "labeling-sessions",
) -> Flask:
    app = Flask(
        __name__,
        template_folder=Path(__file__).parent.parent / "templates",
    )

    if auth:
        require_auth(app)

    store = SessionStore(session_dir)
    logger.info(f"Labeling sessions are stored in {store.directory.resolve()}")

    # One LabelSession per session id, loaded from disk on first use.
    sessions: dict[str, LabelSession] = {}

    def _traversal_order(session_id: str) -> list[int]:
        """Shuffled order, derived from the session id so a resume matches."""
        order = list(range(source.total_images))
        random.Random(session_id).shuffle(order)
        return order

    def _session(session_id: str) -> LabelSession:
        """Get the in-memory session, restoring it from disk when needed."""
        if session_id in sessions:
            return sessions[session_id]

        order = _traversal_order(session_id)
        session = store.load(session_id, order, threshold)
        if session is None:
            session = LabelSession(
                session_id=session_id, threshold=threshold, order=order
            )
            logger.info(f"Started session {session_id} over {len(order)} images")

        sessions[session_id] = session
        return session

    # basename -> glashelder index, built once, used to line up imported entries
    # with this sample even when their stored paths have a different root.
    index_by_name: dict[str, int] = {}

    def _index_by_name() -> dict[str, int]:
        if not index_by_name:
            for index in range(source.total_images):
                path = source.glashelder_path(index)
                if path:
                    index_by_name.setdefault(Path(path).name, index)
        return index_by_name

    def _import_session(data: dict) -> tuple[LabelSession, dict]:
        """Turn an uploaded dataset export into a resumable session."""
        dataset = normalize_dataset(data)
        by_name = _index_by_name()

        resolved = 0
        unresolved: list[str] = []
        labeled: set[int] = set()

        for entry in dataset.entries():
            path = entry.get("glashelder_path") or ""
            index = by_name.get(Path(path).name) if path else None
            if index is None:
                stored = entry.get("glashelder_index")
                if isinstance(stored, int) and 0 <= stored < source.total_images:
                    index = stored
            if index is None:
                unresolved.append(path or str(entry.get("glashelder_index")))
                continue
            # Re-point the entry at this sample's indexing.
            entry["glashelder_index"] = index
            labeled.add(index)
            resolved += 1

        session_id = new_session_id()
        session = LabelSession(
            session_id=session_id,
            threshold=dataset_threshold(dataset, threshold),
            order=_traversal_order(session_id),
            dataset=dataset,
            labeled=labeled,
        )
        sessions[session_id] = session
        store.save(session)

        stats = {
            "entries": len(dataset.entries()),
            "resolved": resolved,
            "unresolved": len(unresolved),
            "already_labeled": len(labeled),
            "remaining": session.remaining,
        }
        logger.info(f"Imported session {session_id}: {stats}")
        if unresolved:
            logger.warning(
                f"{len(unresolved)} imported entries do not match any image in this "
                f"sample (first few: {unresolved[:3]})"
            )
        return session, stats

    def _requested_session() -> LabelSession | None:
        """The session named by the request, or None when it is missing/invalid."""
        session_id = request.values.get("session", "").strip()
        if not session_id or not is_valid_session_id(session_id):
            return None
        return _session(session_id)

    def _context(session: LabelSession, item: MatchItem | None) -> dict:
        base = {
            "session_id": session.session_id,
            "dataset_counts": session.dataset.counts(),
            "total_images": source.total_images,
            "labeled_count": session.labeled_count,
            "remaining": session.remaining,
        }
        if item is None:
            return {
                **base,
                "error": "No images found.",
                "current_index": 0,
                "glashelder_path": "",
                "wikiart_path": "",
                "top_match_index": 0,
                "similarities": [],
                "is_match": False,
                "diff": 0.0,
                "threshold": session.threshold,
                # no item: the page renders an error, so the badge is unused
                "classifier_type": "",
                "features": {},
            }

        wikiart_idx = item.top_indices[0]
        return {
            **base,
            "error": None,
            "current_index": item.glashelder_index,
            "glashelder_path": item.glashelder_path,
            "wikiart_path": source.wikiart_path(wikiart_idx),
            "top_match_index": wikiart_idx,
            "similarities": item.similarities,
            "is_match": item.is_match,
            "diff": item.diff,
            "threshold": item.threshold,
            "classifier_type": item.classifier_type,
            "features": item.features,
        }

    def _exhausted_message(session: LabelSession, want_match: bool) -> str:
        """Say what actually ran out — the old wording read as 'nothing left at all'."""
        if session.remaining == 0:
            return (
                f"All {session.total_images} images in this sample are labeled. "
                "Session complete."
            )
        kind = "matches" if want_match else "negatives"
        return (
            f"No unlabeled {kind} left at the current threshold "
            f"({session.remaining} of {session.total_images} images still unlabeled — "
            f"try the other button or a different threshold)."
        )

    def _render_content(
        session: LabelSession, item: MatchItem | None, want_match: bool
    ) -> str:
        if item is None:
            return render_template(
                "content.html",
                error=_exhausted_message(session, want_match),
                session_id=session.session_id,
                dataset_counts=session.dataset.counts(),
                total_images=source.total_images,
                labeled_count=session.labeled_count,
                remaining=session.remaining,
            )
        return render_template("content.html", **_context(session, item))

    def _find_next(session: LabelSession, want_match: bool) -> MatchItem | None:
        """Next unlabeled image matching the wanted verdict.

        Scans the whole queue starting just after the current position and
        wrapping around, skipping images that already carry a label. Nothing is
        ever stepped past permanently, so the full sample stays reachable.
        """
        n = len(session.order)
        if n == 0:
            return None

        # Already proved there is nothing of this kind left at this threshold.
        if session.known_exhausted(want_match):
            return None

        candidates = [
            pos
            for pos in ((session.queue_position + 1 + k) % n for k in range(n))
            if session.order[pos] not in session.labeled
        ]
        if not candidates:
            return None

        batch_size = max(1, source.batch_size)
        for start in range(0, len(candidates), batch_size):
            chunk_positions = candidates[start : start + batch_size]
            chunk_indices = [session.order[pos] for pos in chunk_positions]
            items = source.classify_many(chunk_indices, session.threshold)
            for pos, item in zip(chunk_positions, items, strict=True):
                if item is not None and item.is_match == want_match:
                    session.queue_position = pos
                    session.current_item = item
                    return item

        session.note_exhausted(want_match)
        return None

    def _advance(session: LabelSession) -> MatchItem | None:
        """Next match if there is one, else next negative."""
        item = _find_next(session, True)
        if item is None:
            item = _find_next(session, False)
        return item

    def _label(session: LabelSession, is_correct: bool) -> str:
        item = cast("MatchItem | None", session.current_item)
        if item is not None:
            wikiart_idx = item.top_indices[0]
            entry = {
                "glashelder_index": item.glashelder_index,
                "glashelder_path": item.glashelder_path,
                "wikiart_index": wikiart_idx,
                "wikiart_path": source.wikiart_path(wikiart_idx),
                "diff": item.diff,
                # the threshold this verdict was decided at, and the verdict itself
                "threshold": item.threshold,
                "is_match": item.is_match,
                "classifier_type": item.classifier_type,
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

            session.labeled.add(item.glashelder_index)
            store.save(session)

        return _render_content(session, _advance(session), True)

    def _missing_session() -> str:
        return render_template(
            "content.html",
            error="This page has no labeling session. Reload the page to start one.",
            dataset_counts=None,
        )

    @app.route("/")
    def index():
        session_id = request.args.get("session", "").strip()
        if not session_id or not is_valid_session_id(session_id):
            session_id = new_session_id()

        session = _session(session_id)
        item = _advance(session)
        if item is None and session.order and session.remaining > 0:
            item = source.classify(session.order[0], session.threshold)
            session.queue_position = 0
            session.current_item = item

        ctx = _context(session, item)
        if item is None:
            ctx["error"] = (
                _exhausted_message(session, True)
                if session.total_images
                else "No images found."
            )
        notice = None
        if request.args.get("imported"):
            notice = (
                f"Imported {session.labeled_count} labeled images into this session — "
                f"{session.remaining} of {session.total_images} still to label."
            )

        return render_template(
            "labeler.html", sessions=store.list_sessions(), notice=notice, **ctx
        )

    @app.route("/new-session", methods=["POST"])
    def new_session():
        session_id = new_session_id()
        _session(session_id)
        response = Response("", status=204)
        response.headers["HX-Redirect"] = f"/?session={session_id}"
        return response

    @app.route("/sessions")
    def list_sessions():
        return {"sessions": store.list_sessions()}

    @app.route("/import-session", methods=["POST"])
    def import_session():
        """Start a session from an uploaded dataset JSON (old or new format)."""
        upload = request.files.get("dataset")
        if upload is None or not upload.filename:
            return _import_error("Choose a dataset JSON file to import.")

        try:
            data = json.load(upload.stream)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _import_error(f"That file is not valid JSON ({e}).")

        if not isinstance(data, dict):
            return _import_error("Expected a JSON object with labeling buckets.")

        try:
            session, stats = _import_session(data)
        except ValueError as e:
            return _import_error(str(e))

        if stats["resolved"] == 0 and stats["entries"] > 0:
            return _import_error(
                f"None of the {stats['entries']} labeled images in that file are part "
                "of the sample this server is serving. Is it the right dataset?"
            )

        response = Response("", status=204)
        response.headers["HX-Redirect"] = f"/?session={session.session_id}&imported=1"
        return response

    def _import_error(message: str) -> str:
        return render_template(
            "content.html", error=f"Import failed: {message}", dataset_counts=None
        )

    @app.route("/next-match", methods=["POST"])
    def next_match():
        session = _requested_session()
        if session is None:
            return _missing_session()
        return _render_content(session, _find_next(session, True), True)

    @app.route("/next-negative", methods=["POST"])
    def next_negative():
        session = _requested_session()
        if session is None:
            return _missing_session()
        return _render_content(session, _find_next(session, False), False)

    @app.route("/set-threshold", methods=["POST"])
    def set_threshold():
        session = _requested_session()
        if session is None:
            return _missing_session()

        new_threshold = request.form.get("threshold", type=float)
        if new_threshold is not None and new_threshold != session.threshold:
            session.set_threshold(new_threshold)
            store.save(session)

        current = cast("MatchItem | None", session.current_item)
        if current:
            item = source.classify(current.glashelder_index, session.threshold)
            session.current_item = item
        else:
            item = None
        return _render_content(session, item, True)

    @app.route("/label-correct", methods=["POST"])
    def label_correct():
        session = _requested_session()
        if session is None:
            return _missing_session()
        return _label(session, True)

    @app.route("/label-incorrect", methods=["POST"])
    def label_incorrect():
        session = _requested_session()
        if session is None:
            return _missing_session()
        return _label(session, False)

    @app.route("/download-dataset")
    def download_dataset():
        session = _requested_session()
        if session is None:
            return "No labeling session given", 400
        return Response(
            session.dataset.to_json(),
            mimetype="application/json",
            headers={
                "Content-Disposition": (
                    f"attachment;filename=labeling_dataset_{session.session_id}.json"
                )
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
