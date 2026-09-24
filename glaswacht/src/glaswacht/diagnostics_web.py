"""Static viewer for glaswacht generation-diagnostics traces.

Reads the JSON traces written by ``Orchestrator.run_diagnostics`` (one file per
model, ``<image-stem>.<model>.json``) and renders them side by side for a single
image: the rating decision distribution, a token-probability heatmap with top-k
alternatives, the forced-rating counterfactuals and the decoding settings.

This module is deliberately a *pure* viewer: it never imports torch and never
runs a model.  Traces are loaded lazily inside the request handlers, so the
module imports cleanly even when ``GW_DIAG_DIR`` is unset and no trace exists.

Environment:
    GW_DIAG_DIR   directory holding the ``*.json`` traces (default ``diagnostics``)
    GW_DIAG_PORT  port to serve on (default 5002, next to the 5001 validator)
    GW_DIAG_LIVE  set to ``1`` to surface the (future) "regenerate from here"
                  hook in the UI.  Unset, the button is not rendered at all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from flask import Flask, redirect, render_template, send_file, url_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_DIR = "diagnostics"
DEFAULT_PORT = 5002

#: Display order of the models, left to right.
MODEL_ORDER = ["lavaguard", "qwen3", "pixtral"]
#: Display order of the rating candidates in the decision distribution.
RATING_ORDER = ["Safe", "Potentially Problematic", "Unsafe"]
#: Tokens below this probability are counted as "low confidence".
LOW_CONFIDENCE = 0.5

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

# Cache of parsed traces, keyed by a directory fingerprint so edits on disk are
# picked up without a restart.  Guarded because app.run() is threaded.
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "traces": []}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def trace_dir() -> Path:
    """Directory holding the trace files (``GW_DIAG_DIR``)."""
    return Path(os.environ.get("GW_DIAG_DIR") or DEFAULT_DIR)


def live_enabled() -> bool:
    """Whether the future live-regeneration hook should be surfaced."""
    return os.environ.get("GW_DIAG_LIVE") == "1"


def server_port() -> int:
    """Port to serve on (``GW_DIAG_PORT``)."""
    raw = os.environ.get("GW_DIAG_PORT")
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid GW_DIAG_PORT=%r, falling back to %d", raw, DEFAULT_PORT)
        return DEFAULT_PORT


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _split_trace_name(path: Path) -> tuple[str, str]:
    """``x.qwen3.json`` -> ``("x", "qwen3")``.  Image stems may contain dots."""
    base = path.name[:-5] if path.name.endswith(".json") else path.name
    stem, _, model = base.rpartition(".")
    if not stem:  # no ".<model>" part at all
        return base, "unknown"
    return stem, model


def _directory_key(directory: Path) -> tuple:
    """Fingerprint of the trace directory, used to invalidate the cache."""
    try:
        files = sorted(directory.glob("*.json"))
    except OSError:
        return ()
    out = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def load_traces() -> list[dict[str, Any]]:
    """Load and cache every trace in ``GW_DIAG_DIR``.

    Never raises: unreadable or malformed files come back as entries with an
    ``error`` key so the page can report them instead of returning a 500.
    """
    directory = trace_dir()
    key = _directory_key(directory)

    with _cache_lock:
        if _cache["key"] == key:
            return _cache["traces"]

    traces: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            stem, model = _split_trace_name(path)
            entry: dict[str, Any] = {
                "path": path,
                "stem": stem,
                "model_key": model,
                "mtime": 0.0,
                "data": None,
                "error": None,
            }
            try:
                entry["mtime"] = path.stat().st_mtime
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    raise ValueError("trace is not a JSON object")
                entry["data"] = data
            except (OSError, ValueError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("Could not read trace %s: %s", path, exc)
            traces.append(entry)
    else:
        logger.info("Trace directory %s does not exist (yet)", directory)

    with _cache_lock:
        _cache["key"] = key
        _cache["traces"] = traces
    logger.info("Loaded %d trace file(s) from %s", len(traces), directory)
    return traces


def _group_by_image(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group traces by image stem, newest group first."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in traces:
        data = entry["data"] or {}
        image = data.get("image") or {}
        # Prefer the stem from the filename: it is what run_diagnostics keys on.
        stem = entry["stem"] or Path(str(image.get("name", ""))).stem
        groups.setdefault(stem, []).append(entry)

    def rank(items: list[dict[str, Any]]) -> tuple[int, float]:
        # Readable groups first, then newest — so an orphaned broken file never
        # hijacks the landing page.
        readable = 1 if any(i["data"] is not None for i in items) else 0
        return (readable, max((i["mtime"] for i in items), default=0.0))

    return dict(sorted(groups.items(), key=lambda kv: rank(kv[1]), reverse=True))


def _model_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    name = _model_name(entry)
    try:
        return (MODEL_ORDER.index(name), name)
    except ValueError:
        return (len(MODEL_ORDER), name)


def _model_name(entry: dict[str, Any]) -> str:
    data = entry["data"] or {}
    model = data.get("model") or {}
    return str(model.get("name") or entry["model_key"])


# --------------------------------------------------------------------------- #
# View models
# --------------------------------------------------------------------------- #


def rating_class(rating: Any) -> str:
    """CSS class for a rating label, matching validate.html's palette."""
    text = str(rating or "").strip().lower()
    if text == "safe":
        return "rating-safe"
    if text == "potentially problematic":
        return "rating-potentially-problematic"
    if text == "unsafe":
        return "rating-unsafe"
    return "rating-unknown"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _width(value: Any) -> float:
    """Bar width in percent, clamped to [0, 100]."""
    try:
        return max(0.0, min(100.0, float(value) * 100.0))
    except (TypeError, ValueError):
        return 0.0


def _heat_color(prob: Any) -> str:
    """Light-theme hsl() ramp: red at p=0 through amber to green at p=1."""
    try:
        p = max(0.0, min(1.0, float(prob)))
    except (TypeError, ValueError):
        p = 0.0
    hue = 4 + 138 * (p**0.75)
    lightness = 91 - 18 * p
    return f"hsl({hue:.0f}, 72%, {lightness:.0f}%)"


def _display_token(text: Any) -> tuple[str, bool]:
    """Make whitespace visible without breaking layout.

    Returns the display string and whether the token is whitespace-only.  The
    container uses ``white-space: pre-wrap``, so the real newline is kept after
    the visible pilcrow and the heatmap keeps the model's own line structure.
    """
    raw = "" if text is None else str(text)
    if raw == "":
        return "∅", True  # empty set: token decodes to nothing
    shown = raw.replace(" ", "·").replace("\t", "→\t")
    shown = shown.replace("\r\n", "\n").replace("\n", "↵\n")
    return shown, raw.strip() == ""


def _build_steps(steps: Any) -> tuple[list[dict[str, Any]], list[dict]]:
    """Spans for the heatmap plus the JSON payload the detail panel reads."""
    spans: list[dict[str, Any]] = []
    payload: list[dict] = []
    if not isinstance(steps, list):
        return spans, payload

    for order, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        index = step.get("i", order)
        prob = step.get("prob")
        shown, is_ws = _display_token(step.get("token"))
        spans.append(
            {
                "i": index,
                "display": shown,
                "color": _heat_color(prob),
                "is_ws": is_ws,
                "is_decision": index == 0,
            }
        )
        alternatives = []
        for alt in step.get("alternatives") or []:
            if not isinstance(alt, dict):
                continue
            alt_shown, _ = _display_token(alt.get("token"))
            alternatives.append(
                {
                    "token_id": alt.get("token_id"),
                    "display": alt_shown,
                    "raw_token": alt.get("raw_token"),
                    "prob": alt.get("prob"),
                    "pct": _pct(alt.get("prob")),
                    "width": _width(alt.get("prob")),
                    "chosen": alt.get("token_id") == step.get("token_id"),
                }
            )
        payload.append(
            {
                "i": index,
                "display": shown,
                "token_id": step.get("token_id"),
                "raw_token": step.get("raw_token"),
                "prob": step.get("prob"),
                "pct": _pct(prob),
                "width": _width(prob),
                "logprob": step.get("logprob"),
                "alternatives": alternatives,
            }
        )
    return spans, payload


def _build_distribution(data: dict[str, Any]) -> dict[str, Any] | None:
    dist = data.get("rating_distribution")
    if not isinstance(dist, dict):
        return None
    raw = dist.get("raw") or {}
    normalized = dist.get("normalized") or {}
    first_tokens = dist.get("candidate_first_tokens") or {}

    labels = list(RATING_ORDER)
    for extra in list(raw) + list(normalized):
        if extra not in labels:
            labels.append(extra)

    rows = []
    for label in labels:
        if label not in raw and label not in normalized:
            continue
        rows.append(
            {
                "label": label,
                "css": rating_class(label),
                "raw_pct": _pct(raw.get(label)),
                "raw_width": _width(raw.get(label)),
                "norm_pct": _pct(normalized.get(label)),
                "norm_width": _width(normalized.get(label)),
                "token_id": first_tokens.get(label),
            }
        )
    return {
        "rows": rows,
        "residual_pct": _pct(dist.get("residual")),
        "residual_width": _width(dist.get("residual")),
        "exact": bool(dist.get("exact")),
    }


def _build_decoding(data: dict[str, Any]) -> dict[str, Any]:
    decoding = data.get("decoding")
    if not isinstance(decoding, dict):
        decoding = {}
    diagnostic = decoding.get("diagnostic") or {}
    production = decoding.get("production") or {}
    keys = list(diagnostic)
    keys += [k for k in production if k not in keys]

    rows = []
    mismatched = []
    for key in keys:
        in_both = key in diagnostic and key in production
        differs = in_both and diagnostic.get(key) != production.get(key)
        if differs:
            mismatched.append(key)
        rows.append(
            {
                "key": key,
                "diagnostic": "—" if key not in diagnostic else diagnostic[key],
                "production": "—" if key not in production else production[key],
                "mismatch": differs,
            }
        )
    return {
        "rows": rows,
        "mismatch": bool(mismatched),
        "mismatch_keys": ", ".join(mismatched),
        "have_production": bool(production),
    }


def _mean_logprob_display(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_counterfactuals(data: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(data.get("counterfactuals") or []):
        if not isinstance(item, dict):
            continue
        parsed = item.get("parsed") or {}
        cf_spans, cf_payload = _build_steps(item.get("steps"))
        mean_logprob = item.get("mean_logprob")
        out.append(
            {
                "kind": item.get("kind", "?"),
                "position": item.get("position"),
                "forced_text": item.get("forced_text", ""),
                "forced_css": rating_class(item.get("forced_text")),
                "forced_token_ids": item.get("forced_token_ids") or [],
                "rating": parsed.get("rating") or "—",
                "rating_css": rating_class(parsed.get("rating")),
                "category": parsed.get("category") or "—",
                "rationale": parsed.get("rationale") or "",
                "generated_text": item.get("generated_text") or "",
                "heatmap_key": f"{model_key}::cf{index}",
                "steps": cf_spans,
                "step_data": cf_payload,
                "step_count": len(cf_payload),
                "mean_logprob": mean_logprob,
                "mean_logprob_display": _mean_logprob_display(mean_logprob),
            }
        )
    return out


def build_model_view(entry: dict[str, Any]) -> dict[str, Any]:
    """Turn one trace file into everything the template needs."""
    name = _model_name(entry)
    if entry["error"]:
        return {
            "name": name,
            "error": entry["error"],
            "file": entry["path"].name,
            "steps": [],
            "step_data": [],
            "step_count": 0,
            "warnings": [],
            "counterfactuals": [],
            "heatmap_entries": [],
        }

    data: dict[str, Any] = entry["data"] or {}
    model = data.get("model") or {}
    parsed = data.get("parsed") or {}
    spans, payload = _build_steps(data.get("steps"))
    timings = data.get("timings_s") or {}
    counterfactuals = _build_counterfactuals(data, name)

    warnings = []
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        warnings.append(
            f"trace schema_version {version!r}, viewer expects {SCHEMA_VERSION}"
        )
    if not spans:
        warnings.append("trace contains no generation steps")

    low_confidence = sum(
        1
        for s in payload
        if isinstance(s.get("prob"), (int, float)) and s["prob"] < LOW_CONFIDENCE
    )

    return {
        "name": name,
        "error": None,
        "file": entry["path"].name,
        "model_id": model.get("model_id", "—"),
        "quantization": model.get("quantization", "—"),
        "parse_model_type": model.get("parse_model_type", "—"),
        "generated_at": data.get("generated_at", "—"),
        "top_k": data.get("top_k", "—"),
        "prefill": data.get("prefill", ""),
        "prompt_token_count": data.get("prompt_token_count", "—"),
        "generated_text": data.get("generated_text", ""),
        "rating": parsed.get("rating") or "—",
        "rating_css": rating_class(parsed.get("rating")),
        "category": parsed.get("category") or "—",
        "rationale": parsed.get("rationale") or "",
        "distribution": _build_distribution(data),
        "heatmap_key": name,
        "steps": spans,
        "step_data": payload,
        "step_count": len(payload),
        "low_confidence": low_confidence,
        "counterfactuals": counterfactuals,
        "decoding": _build_decoding(data),
        "timings": [(k, v) for k, v in timings.items()] if timings else [],
        "warnings": warnings,
        "heatmap_entries": [{"key": name, "steps": payload}]
        + [{"key": cf["heatmap_key"], "steps": cf["step_data"]} for cf in counterfactuals],
    }


def _image_context(views: list[dict[str, Any]], entries: list[dict[str, Any]]) -> dict:
    """Resolve the image these traces describe, tolerating a missing file."""
    name = ""
    path = ""
    for entry in entries:
        image = (entry["data"] or {}).get("image") or {}
        name = name or str(image.get("name") or "")
        path = path or str(image.get("path") or "")
    if not name and path:
        name = Path(path).name
    available = bool(path) and Path(path).is_file()
    return {
        "name": name,
        "path": path,
        "available": available,
        "url": url_for("serve_image", filename=name) if name else "",
        "models": len(views),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def _flatten_heatmaps(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per heatmap shown on the page: each model's main generation
    plus each of its counterfactuals, keyed by the `data-model` id the
    heatmap spans and the click/hover JS both use to look up token data."""
    return [entry for view in views for entry in view.get("heatmap_entries") or []]


@app.route("/")
def index() -> Any:
    """Show the traces for the most recently written image."""
    traces = load_traces()
    groups = _group_by_image(traces)
    directory = trace_dir()

    if not groups:
        return (
            render_template(
                "diagnostics.html",
                empty=True,
                directory=str(directory),
                directory_exists=directory.is_dir(),
                stems=[],
                current_stem=None,
                image=None,
                models=[],
                heatmaps=[],
                live=live_enabled(),
            ),
            200,
        )

    current_stem = next(iter(groups))
    entries = sorted(groups[current_stem], key=_model_sort_key)
    views = [build_model_view(e) for e in entries]

    return render_template(
        "diagnostics.html",
        empty=False,
        directory=str(directory),
        directory_exists=True,
        stems=list(groups),
        current_stem=current_stem,
        image=_image_context(views, entries),
        models=views,
        heatmaps=_flatten_heatmaps(views),
        live=live_enabled(),
    )


@app.route("/traces/<path:stem>")
def show_image(stem: str) -> Any:
    """Show the traces for one specific image stem."""
    traces = load_traces()
    groups = _group_by_image(traces)
    if stem not in groups:
        return redirect(url_for("index"))

    entries = sorted(groups[stem], key=_model_sort_key)
    views = [build_model_view(e) for e in entries]
    return render_template(
        "diagnostics.html",
        empty=False,
        directory=str(trace_dir()),
        directory_exists=True,
        stems=list(groups),
        current_stem=stem,
        image=_image_context(views, entries),
        models=views,
        heatmaps=_flatten_heatmaps(views),
        live=live_enabled(),
    )


@app.route("/image/<path:filename>")
def serve_image(filename: str) -> Any:
    """Serve the source image referenced by a trace.

    Only paths that appear inside a trace file are servable, so an arbitrary
    filename cannot be used to read the filesystem.
    """
    wanted = Path(filename).name
    for entry in load_traces():
        image = (entry["data"] or {}).get("image") or {}
        path = str(image.get("path") or "")
        name = str(image.get("name") or "")
        if wanted not in (name, Path(path).name):
            continue
        candidate = Path(path)
        if candidate.is_file():
            return send_file(candidate)
        logger.info("Image referenced by trace is missing: %s", candidate)
        return ("Image file not found", 404)

    # Fall back to an image sitting next to the traces.
    local = trace_dir() / wanted
    if local.is_file():
        return send_file(local)
    return ("Image not found", 404)


@app.route("/reload")
def reload_traces() -> Any:
    """Drop the cache and re-read the trace directory."""
    with _cache_lock:
        _cache["key"] = None
        _cache["traces"] = []
    load_traces()
    return redirect(url_for("index"))


def main() -> None:
    """Start the diagnostics viewer."""
    port = server_port()
    logger.info(
        "Starting diagnostics viewer on http://localhost:%d (traces: %s)",
        port,
        trace_dir(),
    )
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
