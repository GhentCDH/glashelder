import logging
import os
import subprocess
import sys
import threading
import time
from io import TextIOWrapper
from pathlib import Path

from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from glaswacht.models.utils import get_prompt
from glaswacht.orchestrator import ModelStatus, Orchestrator

_LOG_PATH = Path(__file__).parent / "glaswacht.log"
logger = logging.getLogger("glaswacht")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_PATH)
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s"))
logger.addHandler(_fh)

LOGO = """\
░██████╗░██╗░░░░░░█████╗░░██████╗░██╗░░░░░░░██╗░█████╗░░█████╗░██╗░░██╗████████╗
██╔════╝░██║░░░░░██╔══██╗██╔════╝░██║░░██╗░░██║██╔══██╗██╔══██╗██║░░██║╚══██╔══╝
██║░░██╗░██║░░░░░███████║╚█████╗░░╚██╗████╗██╔╝███████║██║░░╚═╝███████║░░░██║░░░
██║░░╚██╗██║░░░░░██╔══██║░╚═══██╗░░████╔═████║░██╔══██║██║░░██╗██╔══██║░░░██║░░░
╚██████╔╝███████╗██║░░██║██████╔╝░░╚██╔╝░╚██╔╝░██║░░██║╚█████╔╝██║░░██║░░░██║░░░
░╚═════╝░╚══════╝╚═╝░░╚═╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░"""

INFO = "Scripted by @bvercru and @mielpeeters at The Ghent Centre for Digital Humanities (https://www.ghentcdh.ugent.be/).\nRequirements for Glaswacht: Geforce RTX 4090 (24GB VRAM) or better. "


TAXONOMY_PATH = Path(__file__).parent / "models" / "taxonomy.txt"

# Sepia palette
_SEPIA_BRIGHT = "#F5DEB3"  # wheat  - primary text
_SEPIA_MID = "#C8A96E"  # tan    - labels / dim text
_SEPIA_SAFE = "#D4A843"  # amber  - safe
_SEPIA_WARN = "#C87941"  # burnt sienna - potentially problematic
_SEPIA_DANGER = "#8B2500"  # dark rust - unsafe / error
_SEPIA_BORDER = "#8B6914"  # dark gold - borders
_SEPIA_BG = "#1A1008"  # near-black brown - background


def _rating_markup(rating: str) -> str:
    if rating == "Safe":
        return f"[{_SEPIA_SAFE}]{rating}[/{_SEPIA_SAFE}]"
    elif rating == "Potentially Problematic":
        return f"[{_SEPIA_WARN}]{rating}[/{_SEPIA_WARN}]"
    elif rating:
        return f"[{_SEPIA_DANGER}]{rating}[/{_SEPIA_DANGER}]"
    return f"[dim {_SEPIA_MID}]—[/dim {_SEPIA_MID}]"


def _progress_bar(done: int, total: int, width: int = 34) -> str:
    """A block-character bar. With no known total the phase is still running but
    unmeasurable (a model load), so render an empty trough rather than a fake bar."""
    if total <= 0:
        return f"[{_SEPIA_BORDER}]{'░' * width}[/{_SEPIA_BORDER}]"
    fraction = min(max(done / total, 0.0), 1.0)
    filled = int(fraction * width)
    return (
        f"[{_SEPIA_SAFE}]{'█' * filled}[/{_SEPIA_SAFE}]"
        f"[{_SEPIA_BORDER}]{'░' * (width - filled)}[/{_SEPIA_BORDER}]"
    )


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _status_markup(status: ModelStatus) -> str:
    color_map = {
        ModelStatus.CHILLING: _SEPIA_MID,
        ModelStatus.LOADING: _SEPIA_WARN,
        ModelStatus.RUNNING: _SEPIA_BRIGHT,
        ModelStatus.DONE: _SEPIA_SAFE,
        ModelStatus.ERROR: _SEPIA_DANGER,
    }
    color = color_map.get(status, _SEPIA_MID)
    return f"[{color}]{status.value}[/{color}]"


class GlaswachtHeader(Static):
    def render(self):
        return (
            f"[{_SEPIA_BRIGHT}]{LOGO}[/{_SEPIA_BRIGHT}]\n"
            f"[{_SEPIA_MID}]{INFO}[/{_SEPIA_MID}]"
        )


class ModelPanel(Static):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name

    def on_mount(self):
        self.border_title = f" {self._model_name} "

    def update_state(self, state):
        rating_markup = _rating_markup(state.latest_rating)
        image_display = (
            f"[dim]{state.latest_image}[/dim]"
            if state.latest_image
            else f"[{_SEPIA_MID}]—[/{_SEPIA_MID}]"
        )
        task_display = (
            f"[dim]{state.current_task}[/dim]"
            if state.current_task
            else f"[{_SEPIA_MID}]—[/{_SEPIA_MID}]"
        )

        self.update(
            f"[{_SEPIA_MID}]STATUS :[/{_SEPIA_MID}] {_status_markup(state.status)}\n"
            f"[{_SEPIA_MID}]TASK   :[/{_SEPIA_MID}] {task_display}\n"
            f"[{_SEPIA_MID}]IMAGE  :[/{_SEPIA_MID}] {image_display}\n"
            f"[{_SEPIA_MID}]RATING :[/{_SEPIA_MID}] {rating_markup}"
        )


class ContentPanel(Static):
    """Scrollable panel that shows either majority votes or the taxonomy."""

    def show_votes(self, majority_votes: dict):
        if not majority_votes:
            self.update(
                f"[bold {_SEPIA_BRIGHT}][ MAJORITY VOTES ][/bold {_SEPIA_BRIGHT}]\n\n"
                f"[{_SEPIA_MID}]Awaiting results...[/{_SEPIA_MID}]"
            )
            return

        vote_counts = {"Unsafe": 0, "Potentially Problematic": 0, "Safe": 0, "Other": 0}
        for vote in majority_votes.values():
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts["Other"] += 1

        summary_parts = []
        if vote_counts["Safe"] > 0:
            summary_parts.append(
                f"[{_SEPIA_SAFE}]{vote_counts['Safe']} Safe[/{_SEPIA_SAFE}]"
            )
        if vote_counts["Potentially Problematic"] > 0:
            summary_parts.append(
                f"[{_SEPIA_WARN}]{vote_counts['Potentially Problematic']} Potentially Problematic[/{_SEPIA_WARN}]"
            )
        if vote_counts["Unsafe"] > 0:
            summary_parts.append(
                f"[{_SEPIA_DANGER}]{vote_counts['Unsafe']} Unsafe[/{_SEPIA_DANGER}]"
            )
        if vote_counts["Other"] > 0:
            summary_parts.append(
                f"[{_SEPIA_MID}]{vote_counts['Other']} Other[/{_SEPIA_MID}]"
            )

        header = (
            f"[bold {_SEPIA_BRIGHT}][ MAJORITY VOTES ][/bold {_SEPIA_BRIGHT}]"
            f" [{_SEPIA_MID}]({len(majority_votes)} images processed)[/{_SEPIA_MID}]\n"
            f"{'  |  '.join(summary_parts)}\n"
            f"[{_SEPIA_BORDER}]{'─' * 72}[/{_SEPIA_BORDER}]\n"
        )
        vote_lines = [
            f"[{_SEPIA_MID}]{name:<50}[/{_SEPIA_MID}]  {_rating_markup(vote)}"
            for name, vote in majority_votes.items()
        ]
        self.update(header + "\n".join(vote_lines))

    def show_taxonomy(self):
        try:
            text = get_prompt(str(TAXONOMY_PATH))
        except Exception as e:
            text = f"Could not load taxonomy: {e}"
        self.update(
            f"[bold {_SEPIA_BRIGHT}][ TAXONOMY ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]{text}[/{_SEPIA_MID}]"
        )

    def show_webviewer(self, status: str, url: str = ""):
        url_block = (
            f"\n\n[{_SEPIA_MID}]Open in your browser:[/{_SEPIA_MID}] "
            f"[bold {_SEPIA_BRIGHT}]{url}[/bold {_SEPIA_BRIGHT}]"
            if url
            else ""
        )
        self.update(
            f"[bold {_SEPIA_BRIGHT}][ VALIDATE USING WEBVIEWER ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]{status}[/{_SEPIA_MID}]"
            f"{url_block}"
        )

    def show_diagnostics(
        self, image_name: str, status: str, url: str = "", progress: str = ""
    ):
        url_block = (
            f"\n\n[{_SEPIA_MID}]Open in your browser:[/{_SEPIA_MID}] "
            f"[bold {_SEPIA_BRIGHT}]{url}[/bold {_SEPIA_BRIGHT}]"
            if url
            else ""
        )
        progress_block = f"\n\n{progress}" if progress else ""
        self.update(
            f"[bold {_SEPIA_BRIGHT}][ DIAGNOSE ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]IMAGE  :[/{_SEPIA_MID}] [dim]{image_name}[/dim]\n"
            f"[{_SEPIA_MID}]STATUS :[/{_SEPIA_MID}] [{_SEPIA_BRIGHT}]{status}[/{_SEPIA_BRIGHT}]"
            f"{progress_block}"
            f"{url_block}"
        )


class MenuItem(Static):
    selected: reactive[bool] = reactive(False)
    enabled: reactive[bool] = reactive(True)

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self._label = label

    def on_mount(self):
        self._refresh_label()

    def watch_selected(self, _):
        self._refresh_label()

    def watch_enabled(self, _):
        self._refresh_label()

    def _refresh_label(self):
        if not self.enabled:
            self.update(f"[dim {_SEPIA_MID}]  {self._label}[/dim {_SEPIA_MID}]")
        elif self.selected:
            self.update(
                f"[bold {_SEPIA_BG} on {_SEPIA_BRIGHT}]"
                f"  > {self._label} <"
                f"[/bold {_SEPIA_BG} on {_SEPIA_BRIGHT}]"
            )
        else:
            self.update(f"[{_SEPIA_BRIGHT}]  {self._label}[/{_SEPIA_BRIGHT}]")


_MENU_LABELS = ["RUN", "TAXONOMY", "DIAGNOSE", "VALIDATE USING WEBVIEWER", "QUIT"]
_QUIT_INDEX = _MENU_LABELS.index("QUIT")
_WEB_PY_PATH = Path(__file__).parent / "web.py"
_DIAGNOSTICS_PY_PATH = Path(__file__).parent / "diagnostics_web.py"
_DIAG_OUT_DIR = "diagnostics"
_DIAG_PORT = 5002
_WEB_PORT = 5001
# A viewer that dies on startup (missing results file, port already taken) exits
# almost immediately, so re-check shortly after spawning rather than reporting
# success just because Popen returned.
_VIEWER_STARTUP_GRACE = 1.5

# Menu label -> key in MonitorApp._viewers, so activating one menu entry stops the
# viewer belonging to any other.
_VIEWER_KEYS = {"VALIDATE USING WEBVIEWER": "web", "DIAGNOSE": "diagnostics"}


class MonitorApp(App):
    CSS = f"""
    Screen {{
        background: {_SEPIA_BG};
        color: {_SEPIA_MID};
        layout: vertical;
    }}

    #header {{
        border: double {_SEPIA_BORDER};
        padding: 1 2;
        height: auto;
        margin: 1 1 0 1;
    }}

    #models {{
        height: auto;
        margin: 0 0;
    }}

    ModelPanel {{
        border: double {_SEPIA_BORDER};
        border-title-color: {_SEPIA_BRIGHT};
        border-title-style: bold;
        padding: 1 2;
        height: auto;
        width: 1fr;
        margin: 0 1;
        color: {_SEPIA_MID};
    }}

    #content-scroll {{
        border: double {_SEPIA_BORDER};
        height: 1fr;
        margin: 0 1;
        padding: 0 2;
        color: {_SEPIA_MID};
    }}

    #menu {{
        height: auto;
        margin: 0 1 0 1;
    }}

    MenuItem {{
        width: 100%;
        height: 3;
        padding: 0 2;
        border: solid {_SEPIA_BORDER};
        margin: 0;
    }}
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        image_path: str | None = None,
        diag_image: str | None = None,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self._image_path = image_path
        self._diag_image = diag_image
        self.panels: dict[str, ModelPanel] = {}
        self.content_panel: ContentPanel | None = None
        self.menu_widgets: list[MenuItem] = []
        self._selected_index = 0
        self._is_running = False
        self._content_mode = (
            "votes"  # "votes" | "taxonomy" | "webviewer" | "diagnostics"
        )
        self._viewers: dict[str, tuple[subprocess.Popen, TextIOWrapper]] = {}
        self._diag_pending: str | None = None
        self._diag_summary = ""
        self._diag_image_name = ""
        self._diag_started = 0.0

    def compose(self):
        yield GlaswachtHeader(id="header")
        yield Horizontal(id="models")
        yield VerticalScroll(id="content-scroll")
        yield Vertical(id="menu")

    def on_mount(self):
        logger.info("Glaswacht TUI started (image_path=%s)", self._image_path)
        models_container = self.query_one("#models")
        content_container = self.query_one("#content-scroll")
        menu_container = self.query_one("#menu")

        for model_id, state in self.orchestrator.get_state().items():
            panel = ModelPanel(model_name=state.name)
            panel.update_state(state)
            self.panels[model_id] = panel
            models_container.mount(panel)

        self.content_panel = ContentPanel()
        self.content_panel.show_votes({})
        content_container.mount(self.content_panel)

        for i, label in enumerate(_MENU_LABELS):
            item = MenuItem(label)
            item.selected = i == 0
            self.menu_widgets.append(item)
            menu_container.mount(item)

        self.set_interval(0.1, self._update_display)

    def on_key(self, event):
        key = event.key
        if key == "up":
            self._move_selection(-1)
        elif key == "down":
            self._move_selection(1)
        elif key in ("enter", "return"):
            self._activate()

    def _move_selection(self, delta: int):
        n = len(_MENU_LABELS)
        new = (self._selected_index + delta) % n
        for _ in range(n):
            if self.menu_widgets[new].enabled:
                self._diag_pending = None
                self.menu_widgets[self._selected_index].selected = False
                self._selected_index = new
                self.menu_widgets[new].selected = True
                return
            new = (new + delta) % n

    def _menu_item(self, label: str) -> MenuItem:
        return self.menu_widgets[_MENU_LABELS.index(label)]

    def _set_menu_enabled(self, enabled: bool):
        """Enable/disable every menu entry except QUIT, which always stays usable."""
        for label in _MENU_LABELS:
            if label != "QUIT":
                self._menu_item(label).enabled = enabled

    def _select_quit(self):
        self.menu_widgets[self._selected_index].selected = False
        self._selected_index = _QUIT_INDEX
        self.menu_widgets[_QUIT_INDEX].selected = True

    def _start_viewer(
        self, key: str, script: Path, env: dict[str, str] | None = None
    ) -> subprocess.Popen:
        running = self._viewers.get(key)
        if running is not None:
            if running[0].poll() is None:
                return running[0]
            self._stop_viewer(key)
        log_fh = open(_LOG_PATH, "a")
        process = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log_fh,
            stderr=log_fh,
            env={**os.environ, **env} if env else None,
        )
        self._viewers[key] = (process, log_fh)
        return process

    def _stop_viewer(self, key: str | None = None):
        for name in list(self._viewers) if key is None else [key]:
            viewer = self._viewers.pop(name, None)
            if viewer is None:
                continue
            process, log_fh = viewer
            if process.poll() is None:
                process.terminate()
                process.wait()
            log_fh.close()

    def _activate(self):
        label = _MENU_LABELS[self._selected_index]
        for key in list(self._viewers):
            if key != _VIEWER_KEYS.get(label):
                self._stop_viewer(key)
        if label == "RUN":
            self._start_run()
        elif label == "TAXONOMY":
            self._show_taxonomy()
        elif label == "DIAGNOSE":
            self._start_diagnostics()
        elif label == "VALIDATE USING WEBVIEWER":
            self._start_webviewer()
        elif label == "QUIT":
            self.exit()

    def _start_run(self):
        if self._is_running:
            return
        logger.info("RUN requested, path=%s", self._image_path)
        self._is_running = True
        self._content_mode = "votes"
        self._set_menu_enabled(False)
        self._select_quit()

        def _run():
            if self._image_path is None:
                self.notify(
                    "No --path provided. Use: glaswacht tui --path <dir>",
                    severity="error",
                )
                self.call_from_thread(self._on_run_complete)
                return
            self.orchestrator.run_all(self._image_path)
            self.call_from_thread(self._on_run_complete)

        threading.Thread(target=_run, daemon=True).start()

    def _on_run_complete(self):
        logger.info("Run complete")
        self._is_running = False
        self._set_menu_enabled(True)

    def _show_taxonomy(self):
        self._content_mode = "taxonomy"
        if self.content_panel:
            self.content_panel.show_taxonomy()

    def _start_webviewer(self):
        process = self._start_viewer("web", _WEB_PY_PATH)
        logger.info("Webviewer started (pid=%s)", process.pid)
        self._content_mode = "webviewer"
        if self.content_panel:
            self.content_panel.show_webviewer("Starting the validation web server...")
        self.set_timer(_VIEWER_STARTUP_GRACE, self._check_webviewer)

    def _check_webviewer(self):
        viewer = self._viewers.get("web")
        if viewer is None or self._content_mode != "webviewer":
            return
        process, _ = viewer
        if process.poll() is None:
            if self.content_panel:
                self.content_panel.show_webviewer(
                    "The validation web server is running.",
                    f"http://localhost:{_WEB_PORT}",
                )
            return

        code = process.returncode
        self._stop_viewer("web")
        logger.error("Webviewer exited with code %s", code)
        if self.content_panel:
            self.content_panel.show_webviewer(
                f"Server exited immediately (code {code}). "
                f"See {_LOG_PATH} for the traceback.\n\n"
                "Most likely no results CSV was found. Start it explicitly with:\n"
                "  glaswacht serve --csv-path <file> --images-dir <dir>"
            )

    def _resolve_diag_image(self) -> str | None:
        """Pick the single image to diagnose: --diag-image, else the model panels'
        latest image, else the alphabetically first image in --path."""
        if self._diag_image:
            return self._diag_image
        if self._image_path is None:
            return None
        from glaswacht.models.utils import get_images

        try:
            candidates = sorted(get_images(self._image_path))
        except ValueError as exc:
            logger.warning("Could not list images: %s", exc)
            return None
        if not candidates:
            return None

        latest = next(
            (
                s.latest_image
                for s in self.orchestrator.get_state().values()
                if s.latest_image
            ),
            "",
        )
        by_name = {os.path.basename(path): path for path in candidates}
        return by_name.get(latest, candidates[0])

    def _start_diagnostics(self):
        if self._is_running:
            return
        image = self._resolve_diag_image()
        self._content_mode = "diagnostics"
        if image is None:
            if self.content_panel:
                self.content_panel.show_diagnostics(
                    "—",
                    "No image. Use: glaswacht tui --path <dir> [--diag-image <file>]",
                )
            return

        # A diagnostics pass loads all three models, so confirm the resolved image
        # first instead of committing to it on a single keypress.
        if self._diag_pending != image:
            self._diag_pending = image
            if self.content_panel:
                self.content_panel.show_diagnostics(
                    os.path.basename(image),
                    "Press ENTER again to start, or move the selection to cancel.",
                )
            return
        self._diag_pending = None

        logger.info("DIAGNOSE requested, image=%s", image)
        self._is_running = True
        self._diag_image_name = os.path.basename(image)
        self._diag_started = time.monotonic()
        self._set_menu_enabled(False)
        self._select_quit()
        if self.content_panel:
            self.content_panel.show_diagnostics(
                self._diag_image_name, "Starting diagnostics across all models..."
            )

        def _run():
            try:
                traces = self.orchestrator.run_diagnostics(image, _DIAG_OUT_DIR)
            except Exception as exc:
                logger.error("Diagnostics failed", exc_info=True)
                self.call_from_thread(
                    self._on_diagnostics_complete, image, [], f"Failed: {exc}"
                )
                return
            self.call_from_thread(self._on_diagnostics_complete, image, traces, "")

        threading.Thread(target=_run, daemon=True).start()

    def _on_diagnostics_complete(self, image: str, traces: list, error: str):
        logger.info("Diagnostics complete (%d traces)", len(traces))
        self._is_running = False
        self._set_menu_enabled(True)
        name = os.path.basename(image)

        if error or not traces:
            if self.content_panel:
                self.content_panel.show_diagnostics(
                    name, error or "No traces were written."
                )
            return

        process = self._start_viewer(
            "diagnostics",
            _DIAGNOSTICS_PY_PATH,
            env={
                "GW_DIAG_DIR": str(Path(_DIAG_OUT_DIR).resolve()),
                "GW_DIAG_PORT": str(_DIAG_PORT),
            },
        )
        logger.info("Diagnostics viewer started (pid=%s)", process.pid)
        self._diag_summary = f"{len(traces)} trace(s) written to {_DIAG_OUT_DIR}/"
        if self.content_panel:
            self.content_panel.show_diagnostics(
                name, f"{self._diag_summary} — starting viewer..."
            )
        self.set_timer(
            _VIEWER_STARTUP_GRACE, lambda: self._check_diagnostics_viewer(name)
        )

    def _check_diagnostics_viewer(self, name: str):
        viewer = self._viewers.get("diagnostics")
        if viewer is None or self._content_mode != "diagnostics":
            return
        process, _ = viewer
        if process.poll() is None:
            if self.content_panel:
                self.content_panel.show_diagnostics(
                    name, self._diag_summary, url=f"http://localhost:{_DIAG_PORT}"
                )
            return

        code = process.returncode
        self._stop_viewer("diagnostics")
        logger.error("Diagnostics viewer exited with code %s", code)
        if self.content_panel:
            self.content_panel.show_diagnostics(
                name,
                f"{self._diag_summary}\nViewer exited immediately (code {code}). "
                f"See {_LOG_PATH} for the traceback.",
            )

    def _render_diagnostics_progress(self):
        state = self.orchestrator.get_state()
        if not state or self.content_panel is None:
            return
        finished = sum(1 for s in state.values() if s.status is ModelStatus.DONE)
        current = next(
            (
                s
                for s in state.values()
                if s.status in (ModelStatus.LOADING, ModelStatus.RUNNING)
            ),
            None,
        )
        elapsed = _elapsed(time.monotonic() - self._diag_started)

        if current is None:
            status = f"Writing traces... ({finished}/{len(state)} models)"
            bar, detail = _progress_bar(0, 0), elapsed
        else:
            status = (
                f"{current.name} — {current.phase or 'working'} "
                f"({finished + 1}/{len(state)})"
            )
            bar = _progress_bar(current.progress_done, current.progress_total)
            detail = (
                f"{current.progress_done}/{current.progress_total} tokens   {elapsed}"
                if current.progress_total
                else elapsed
            )

        self.content_panel.show_diagnostics(
            self._diag_image_name,
            status,
            progress=f"{bar}  [{_SEPIA_MID}]{detail}[/{_SEPIA_MID}]",
        )

    def _update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)

        if self._content_mode == "diagnostics" and self._is_running:
            self._render_diagnostics_progress()
            return

        if self._content_mode == "votes":
            majority_votes = self.orchestrator.get_majority_votes()
            if self.content_panel:
                self.content_panel.show_votes(majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run(mouse=False)
    app._stop_viewer()
