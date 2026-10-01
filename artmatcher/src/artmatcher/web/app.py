import json
import logging
import random
from pathlib import Path
from typing import cast
from urllib.parse import quote

from flask import Flask, Response, redirect, render_template, request, send_file

from common import require_auth
from common.images import is_url

from .session import LabelSession
from .source import ClassificationSource, MatchItem
from .store import (
    SessionStore,
    dataset_threshold,
    is_valid_session_id,
    normalize_dataset,
    slugify_session_name,
)

logger = logging.getLogger(__name__)


def create_app(
    source: ClassificationSource,
    *,
    threshold: float,
    auth: bool,
    session_dir: str | Path = "labeling-sessions",
    subset: list[int] | None = None,
) -> Flask:
    """The labeling web app over `source`.

    With `subset` (glashelder indices), every session only walks through those
    images; they keep their indices in the full set, so labels stay compatible.
    """
    sample = sorted(set(subset)) if subset is not None else range(source.total_images)
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
        order = list(sample)
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

    def _import_session(data: dict, session_id: str) -> tuple[LabelSession, dict]:
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

    def _queue(session: LabelSession) -> dict:
        """Where the session is in its (random, fixed) order, for the arrow."""
        position = (
            session.scan_position
            if session.scan_position is not None
            else session.queue_position
        )
        return {"queue_position": position, "queue_size": len(session.order)}

    def _context(session: LabelSession, item: MatchItem | None) -> dict:
        base = {
            "session_id": session.session_id,
            "dataset_counts": session.dataset.counts(),
            "total_images": source.total_images,
            "labeled_count": session.labeled_count,
            "remaining": session.remaining,
            "labeled_indices": session.labeled,
            **_queue(session),
        }
        if item is None:
            return {
                **base,
                "error": "No images found.",
                "current_index": 0,
                "glashelder_path": "",
                "artwork_path": "",
                "top_match_index": 0,
                "artwork": None,
                "alternatives": [],
                "similarities": [],
                "is_match": False,
                "diff": 0.0,
                "threshold": session.threshold,
                # no item: the page renders an error, so the badge is unused
                "classifier_type": "",
                "features": {},
            }

        artwork_idx = item.best_index
        # the other top matches, in CLIP order, for a look at what else came close
        alternatives = [
            {
                "index": index,
                "rank": rank + 1,
                "similarity": similarity,
                "artwork_path": source.artwork_path(index),
                "artwork": source.artwork(index),
            }
            for rank, (index, similarity) in enumerate(
                zip(item.top_indices, item.top_similarities, strict=False)
            )
            if rank != item.rank
        ][:4]
        return {
            **base,
            "error": None,
            "current_index": item.glashelder_index,
            "glashelder_path": item.glashelder_path,
            "artwork_path": source.artwork_path(artwork_idx),
            "top_match_index": artwork_idx,
            "artwork": source.artwork(artwork_idx),
            "alternatives": alternatives,
            "match_rank": item.rank + 1,
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

    def _progress_oob(session: LabelSession) -> str:
        """The session bar's counter, swapped out-of-band.

        The bar sits outside #content, so without this it would keep showing
        the counts from when the page was first loaded.
        """
        return render_template(
            "session_progress.html",
            oob=True,
            labeled_count=session.labeled_count,
            queue_size=session.total_images,
            remaining=session.remaining,
        )

    def _render_content(
        session: LabelSession, item: MatchItem | None, want_match: bool
    ) -> str:
        if item is None:
            body = render_template(
                "content.html",
                error=_exhausted_message(session, want_match),
                session_id=session.session_id,
                dataset_counts=session.dataset.counts(),
                total_images=source.total_images,
                labeled_count=session.labeled_count,
                remaining=session.remaining,
                **_queue(session),
            )
        else:
            body = render_template("content.html", **_context(session, item))
        return body + _progress_oob(session)

    def _find_next(
        session: LabelSession, want_match: bool, seq: int | None = None
    ) -> MatchItem | None:
        """Next unlabeled image matching the wanted verdict.

        Scans the whole queue starting just after the current position and
        wrapping around, skipping images that already carry a label. Nothing is
        ever stepped past permanently, so the full sample stays reachable.

        `seq` makes the scan abandonable: when a newer request has arrived the
        scan stops at the next chunk boundary and touches no session state.
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
        try:
            for start in range(0, len(candidates), batch_size):
                # A newer click has taken over: stop burning CPU on a result whose
                # response the browser has already thrown away.
                if not session.is_current(seq):
                    logger.info(
                        f"Abandoning {'match' if want_match else 'negative'} search in "
                        f"{session.session_id}: superseded by a newer request"
                    )
                    return None

                chunk_positions = candidates[start : start + batch_size]
                # for the arrow on the page, which polls this while we search
                session.scan_position = chunk_positions[0]
                chunk_indices = [session.order[pos] for pos in chunk_positions]
                items = source.classify_many(chunk_indices, session.threshold)
                for pos, item in zip(chunk_positions, items, strict=True):
                    if item is not None and item.is_match == want_match:
                        # Only the newest request may move the cursor.
                        if session.is_current(seq):
                            session.queue_position = pos
                            session.current_item = item
                        return item

            if session.is_current(seq):
                session.note_exhausted(want_match)
            return None
        finally:
            if session.is_current(seq):
                session.scan_position = None

    def _advance(session: LabelSession, seq: int | None = None) -> MatchItem | None:
        """Next image of the kind being worked on, falling back to the other.

        `want_match` follows the last Next Match / Next Negative press, so
        labeling a negative serves the next negative instead of jumping back
        to matches. The fallback keeps the session going once one kind runs out.
        """
        item = _find_next(session, session.want_match, seq)
        if item is None and session.is_current(seq):
            item = _find_next(session, not session.want_match, seq)
        return item

    def _label(session: LabelSession, is_correct: bool) -> str:
        # The label itself always commits -- it is the user's actual work. Only
        # the search for the next image afterwards is abandonable.
        item = cast("MatchItem | None", session.current_item)
        if item is not None:
            artwork_idx = item.best_index
            entry = {
                "glashelder_index": item.glashelder_index,
                "glashelder_path": item.glashelder_path,
                "artwork_idx": artwork_idx,
                "artwork_path": source.artwork_path(artwork_idx),
                # which CLIP match was labeled (0 = the best); its similarities
                # below are the ones the classifier saw for it
                "rank": item.rank,
                "diff": item.diff,
                # the threshold this verdict was decided at, and the verdict itself
                "threshold": item.threshold,
                "is_match": item.is_match,
                "classifier_type": item.classifier_type,
                "similarities": item.similarities,
            }
            # an image opened again by its number replaces its label, rather than
            # adding a second one that would weigh double in training
            for bucket in (
                session.dataset.positives,
                session.dataset.negatives,
                session.dataset.false_positives,
                session.dataset.false_negatives,
            ):
                bucket[:] = [
                    old
                    for old in bucket
                    if old.get("glashelder_index") != item.glashelder_index
                ]
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

        seq = session.begin_request()
        return _render_content(session, _advance(session, seq), session.want_match)

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
            # Sessions are always named by a person, so there is nothing to
            # fall back to: ask for a name instead of inventing one.
            return render_template(
                "start.html",
                sessions=store.list_sessions(),
                total_images=len(sample),
                error=request.args.get("error"),
            )

        session = _session(session_id)
        item = _advance(session, session.begin_request())
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
            "labeler.html",
            sessions=store.list_sessions(),
            notice=notice,
            page_error=request.args.get("error"),
            **ctx,
        )

    def _requested_name(field: str = "name") -> tuple[str, str | None]:
        """Slugify a name typed by a person; returns (id, error)."""
        raw = request.form.get(field, "")
        session_id = slugify_session_name(raw)
        if not session_id:
            return "", (
                "Give the session a name (letters, digits, dashes and underscores)."
            )
        if store.exists(session_id):
            return (
                "",
                f"A session named {session_id!r} already exists — resume it instead.",
            )
        return session_id, None

    @app.route("/new-session", methods=["POST"])
    def new_session():
        session_id, error = _requested_name()
        if error:
            # Stay where they were, rather than dumping them out of a session
            # they were in the middle of, just because a name was rejected.
            came_from = request.form.get("session", "").strip()
            keep = (
                f"session={came_from}&"
                if came_from and is_valid_session_id(came_from)
                else ""
            )
            return _redirect_to(f"/?{keep}error={quote(error)}")

        # Created and opened in one step: the new session becomes the live one.
        session = _session(session_id)
        store.save(session)
        return _redirect_to(f"/?session={session_id}")

    @app.route("/rename-session", methods=["POST"])
    def rename_session():
        session = _requested_session()
        if session is None:
            return _missing_session()

        new_id, error = _requested_name("new_name")
        if error:
            return _render_content_with_error(session, f"Rename failed: {error}")

        old_id = session.session_id
        try:
            store.rename(old_id, new_id)
        except ValueError as e:
            return _render_content_with_error(session, f"Rename failed: {e}")

        sessions.pop(old_id, None)
        session.session_id = new_id
        sessions[new_id] = session
        store.save(session)
        return _redirect_to(f"/?session={new_id}")

    def _redirect_to(location: str) -> Response:
        """HTMX-aware redirect: a header for hx requests, a 302 otherwise."""
        if request.headers.get("HX-Request"):
            response = Response("", status=204)
            response.headers["HX-Redirect"] = location
            return response
        response = Response("", status=302)
        response.headers["Location"] = location
        return response

    def _render_content_with_error(session: LabelSession, message: str) -> str:
        return render_template(
            "content.html",
            error=message,
            session_id=session.session_id,
            dataset_counts=session.dataset.counts(),
            total_images=source.total_images,
            labeled_count=session.labeled_count,
            remaining=session.remaining,
            **_queue(session),
        ) + _progress_oob(session)

    @app.route("/queue-position")
    def queue_position():
        """Where the session is in its order, polled by the page during a search."""
        session = _requested_session()
        if session is None:
            return {"error": "no labeling session given"}, 400
        return _queue(session)

    @app.route("/sessions")
    def list_sessions():
        return {"sessions": store.list_sessions()}

    @app.route("/import-session", methods=["POST"])
    def import_session():
        """Start a session from an uploaded dataset JSON (old or new format)."""
        upload = request.files.get("dataset")
        if upload is None or not upload.filename:
            return _import_error("Choose a dataset JSON file to import.")

        session_id, name_error = _requested_name()
        if name_error:
            return _import_error(name_error)

        try:
            data = json.load(upload.stream)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _import_error(f"That file is not valid JSON ({e}).")

        if not isinstance(data, dict):
            return _import_error("Expected a JSON object with labeling buckets.")

        try:
            session, stats = _import_session(data, session_id)
        except ValueError as e:
            return _import_error(str(e))

        if stats["resolved"] == 0 and stats["entries"] > 0:
            sessions.pop(session_id, None)
            return _import_error(
                f"None of the {stats['entries']} labeled images in that file are part "
                "of the sample this server is serving. Is it the right dataset?"
            )

        return _redirect_to(f"/?session={session.session_id}&imported=1")

    def _import_error(message: str):
        """Report an import problem, as a fragment or a redirect as appropriate."""
        if request.headers.get("HX-Request"):
            return render_template(
                "content.html", error=f"Import failed: {message}", dataset_counts=None
            )
        return _redirect_to(f"/?error={quote(f'Import failed: {message}')}")

    @app.route("/next-match", methods=["POST"])
    def next_match():
        session = _requested_session()
        if session is None:
            return _missing_session()
        seq = session.begin_request()
        session.want_match = True
        store.save(session)
        return _render_content(session, _find_next(session, True, seq), True)

    @app.route("/next-negative", methods=["POST"])
    def next_negative():
        session = _requested_session()
        if session is None:
            return _missing_session()
        seq = session.begin_request()
        session.want_match = False
        store.save(session)
        return _render_content(session, _find_next(session, False, seq), False)

    @app.route("/go-to-image", methods=["POST"])
    def go_to_image():
        """Match one image chosen by its number, and continue from there.

        The cursor moves to that image's place in the session's order, so the
        next search (and the arrow under the dataset row) carry on from it.
        """
        session = _requested_session()
        if session is None:
            return _missing_session()
        seq = session.begin_request()

        number = request.form.get("index", type=int)
        if number is None or not 1 <= number <= source.total_images:
            return _render_content_with_error(
                session, f"There is no image {number}: pick 1 to {source.total_images}."
            )

        index = number - 1
        if not session.includes(index):
            return _render_content_with_error(
                session, f"Image {number} is not part of the subset this server labels."
            )
        item = source.classify(index, session.threshold)
        if session.is_current(seq):
            session.queue_position = session.order.index(index)
            session.current_item = item
        return _render_content(session, item, session.want_match)

    @app.route("/set-threshold", methods=["POST"])
    def set_threshold():
        session = _requested_session()
        if session is None:
            return _missing_session()

        # Supersedes any in-flight search, whose verdicts were decided at the
        # old threshold and would land on screen looking authoritative.
        seq = session.begin_request()

        new_threshold = request.form.get("threshold", type=float)
        if new_threshold is not None and new_threshold != session.threshold:
            session.set_threshold(new_threshold)
            store.save(session)

        current = cast("MatchItem | None", session.current_item)
        if current:
            item = source.classify(current.glashelder_index, session.threshold)
            if session.is_current(seq):
                session.current_item = item
        else:
            item = None
        return _render_content(session, item, session.want_match)

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

    @app.route("/orb-overlay/<int:index>/<int:artwork_index>")
    def serve_orb_overlay(index: int, artwork_index: int):
        """The glashelder image warped onto the artwork, half transparent.

        Computed on request (only when the labeler opens it), with the same ORB
        matching the features use.
        """
        import cv2

        from ..ml_classifier import orb_overlay

        query, reference = (
            source.glashelder_path(index),
            source.artwork_path(artwork_index),
        )
        if not query or not reference:
            return "Image not found", 404
        try:
            image = orb_overlay(query, reference)
        except Exception as e:
            logger.warning(f"ORB overlay of {query} on {reference} failed: {e}")
            return f"Could not load the images ({e})", 502
        if image is None:
            return "ORB found no homography between these images", 404
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(jpeg.tobytes(), mimetype="image/jpeg")

    @app.route("/artwork-image/<int:index>")
    def serve_artwork_image(index: int):
        reference = source.artwork_path(index)
        if not reference:
            return "Image not found", 404
        if is_url(reference):
            # let the browser fetch it straight from the source, so the labeler
            # never waits on this server doing IO
            return redirect(reference)
        return send_file(reference)

    return app
