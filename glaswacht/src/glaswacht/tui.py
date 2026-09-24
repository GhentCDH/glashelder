import threading
from pathlib import Path

from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static

from glaswacht.models.utils import get_prompt
from glaswacht.orchestrator import Orchestrator
from glaswacht.tui_common import (
    LOG_PATH,
    SEPIA_BRIGHT,
    SEPIA_DANGER,
    SEPIA_MID,
    SEPIA_SAFE,
    SEPIA_WARN,
    VIEWER_STARTUP_GRACE,
    GlaswachtHeader,
    MenuItem,
    ModelPanel,
    base_css,
    logger,
    rating_markup,
    start_viewer,
    stop_viewer,
)

TAXONOMY_PATH = Path(__file__).parent / "models" / "taxonomy.txt"


class ContentPanel(Static):
    """Scrollable panel that shows either majority votes or the taxonomy."""

    def show_votes(self, majority_votes: dict):
        if not majority_votes:
            self.update(
                f"[bold {SEPIA_BRIGHT}][ MAJORITY VOTES ][/bold {SEPIA_BRIGHT}]\n\n"
                f"[{SEPIA_MID}]Awaiting results...[/{SEPIA_MID}]"
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
                f"[{SEPIA_SAFE}]{vote_counts['Safe']} Safe[/{SEPIA_SAFE}]"
            )
        if vote_counts["Potentially Problematic"] > 0:
            summary_parts.append(
                f"[{SEPIA_WARN}]{vote_counts['Potentially Problematic']} Potentially Problematic[/{SEPIA_WARN}]"
            )
        if vote_counts["Unsafe"] > 0:
            summary_parts.append(
                f"[{SEPIA_DANGER}]{vote_counts['Unsafe']} Unsafe[/{SEPIA_DANGER}]"
            )
        if vote_counts["Other"] > 0:
            summary_parts.append(
                f"[{SEPIA_MID}]{vote_counts['Other']} Other[/{SEPIA_MID}]"
            )

        header = (
            f"[bold {SEPIA_BRIGHT}][ MAJORITY VOTES ][/bold {SEPIA_BRIGHT}]"
            f" [{SEPIA_MID}]({len(majority_votes)} images processed)[/{SEPIA_MID}]\n"
            f"{'  |  '.join(summary_parts)}\n"
            f"[{SEPIA_MID}]{'─' * 72}[/{SEPIA_MID}]\n"
        )
        vote_lines = [
            f"[{SEPIA_MID}]{name:<50}[/{SEPIA_MID}]  {rating_markup(vote)}"
            for name, vote in majority_votes.items()
        ]
        self.update(header + "\n".join(vote_lines))

    def show_taxonomy(self):
        try:
            text = get_prompt(str(TAXONOMY_PATH))
        except Exception as e:
            text = f"Could not load taxonomy: {e}"
        self.update(
            f"[bold {SEPIA_BRIGHT}][ TAXONOMY ][/bold {SEPIA_BRIGHT}]\n\n"
            f"[{SEPIA_MID}]{text}[/{SEPIA_MID}]"
        )

    def show_webviewer(self, status: str, url: str = ""):
        url_block = (
            f"\n\n[{SEPIA_MID}]Open in your browser:[/{SEPIA_MID}] "
            f"[bold {SEPIA_BRIGHT}]{url}[/bold {SEPIA_BRIGHT}]"
            if url
            else ""
        )
        self.update(
            f"[bold {SEPIA_BRIGHT}][ VALIDATE USING WEBVIEWER ][/bold {SEPIA_BRIGHT}]\n\n"
            f"[{SEPIA_MID}]{status}[/{SEPIA_MID}]"
            f"{url_block}"
        )


_MENU_LABELS = ["RUN", "TAXONOMY", "VALIDATE USING WEBVIEWER", "QUIT"]
_QUIT_INDEX = _MENU_LABELS.index("QUIT")
_WEB_PY_PATH = Path(__file__).parent / "web.py"
_WEB_PORT = 5001

# Menu label -> key in MonitorApp._viewers, so activating one menu entry stops the
# viewer belonging to any other.
_VIEWER_KEYS = {"VALIDATE USING WEBVIEWER": "web"}


class MonitorApp(App):
    CSS = base_css()

    def __init__(self, orchestrator: Orchestrator, image_path: str | None = None):
        super().__init__()
        self.orchestrator = orchestrator
        self._image_path = image_path
        self.panels: dict[str, ModelPanel] = {}
        self.content_panel: ContentPanel | None = None
        self.menu_widgets: list[MenuItem] = []
        self._selected_index = 0
        self._is_running = False
        self._content_mode = "votes"  # "votes" | "taxonomy" | "webviewer"
        self._viewers: dict[str, tuple] = {}

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

    def stop_viewers(self):
        stop_viewer(self._viewers)

    def _activate(self):
        label = _MENU_LABELS[self._selected_index]
        for key in list(self._viewers):
            if key != _VIEWER_KEYS.get(label):
                stop_viewer(self._viewers, key)
        if label == "RUN":
            self._start_run()
        elif label == "TAXONOMY":
            self._show_taxonomy()
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
        process = start_viewer(self._viewers, "web", _WEB_PY_PATH)
        logger.info("Webviewer started (pid=%s)", process.pid)
        self._content_mode = "webviewer"
        if self.content_panel:
            self.content_panel.show_webviewer("Starting the validation web server...")
        self.set_timer(VIEWER_STARTUP_GRACE, self._check_webviewer)

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
        stop_viewer(self._viewers, "web")
        logger.error("Webviewer exited with code %s", code)
        if self.content_panel:
            self.content_panel.show_webviewer(
                f"Server exited immediately (code {code}). "
                f"See {LOG_PATH} for the traceback.\n\n"
                "Most likely no results CSV was found. Start it explicitly with:\n"
                "  glaswacht serve --csv-path <file> --images-dir <dir>"
            )

    def _update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)

        if self._content_mode == "votes":
            majority_votes = self.orchestrator.get_majority_votes()
            if self.content_panel:
                self.content_panel.show_votes(majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run(mouse=False)
    app.stop_viewers()
