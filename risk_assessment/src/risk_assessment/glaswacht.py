import logging
import subprocess
import sys
import threading
from pathlib import Path

from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from risk_assessment.models.utils import get_prompt
from risk_assessment.orchestrator import ModelStatus, Orchestrator

_LOG_PATH = Path(__file__).parent / "glaswacht.log"
logger = logging.getLogger("glaswacht")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_PATH)
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s"))
logger.addHandler(_fh)
# Also capture logs from the risk_assessment package (orchestrator, models, etc.)
_ra_logger = logging.getLogger("risk_assessment")
_ra_logger.setLevel(logging.DEBUG)
_ra_logger.addHandler(_fh)

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

    def show_webviewer(self):
        self.update(
            f"[bold {_SEPIA_BRIGHT}][ VALIDATE USING WEBVIEWER ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]The validation web server is running.[/{_SEPIA_MID}]\n\n"
            f"[{_SEPIA_MID}]Open in your browser:[/{_SEPIA_MID}] "
            f"[bold {_SEPIA_BRIGHT}]http://localhost:5001[/bold {_SEPIA_BRIGHT}]"
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


_MENU_LABELS = ["RUN", "TAXONOMY", "VALIDATE USING WEBVIEWER", "QUIT"]
_WEB_PY_PATH = Path(__file__).parent / "web.py"


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
        self._web_process: subprocess.Popen | None = None

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

    def _stop_webviewer(self):
        if self._web_process and self._web_process.poll() is None:
            self._web_process.terminate()
            self._web_process = None

    def _activate(self):
        label = _MENU_LABELS[self._selected_index]
        if label != "VALIDATE USING WEBVIEWER":
            self._stop_webviewer()
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
        # Disable RUN, TAXONOMY and VALIDATE while running
        self.menu_widgets[0].enabled = False
        self.menu_widgets[1].enabled = False
        self.menu_widgets[2].enabled = False
        # Move selection to QUIT (index 3) so the user can still quit
        self.menu_widgets[self._selected_index].selected = False
        self._selected_index = 3
        self.menu_widgets[3].selected = True

        def _run():
            if self._image_path is None:
                self.notify(
                    "No --path provided. Use: risk-assessment glaswacht --path <dir>",
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
        self.menu_widgets[0].enabled = True
        self.menu_widgets[1].enabled = True
        self.menu_widgets[2].enabled = True

    def _show_taxonomy(self):
        self._content_mode = "taxonomy"
        self.content_panel.show_taxonomy()

    def _start_webviewer(self):
        # Only launch if not already running
        if self._web_process is None or self._web_process.poll() is not None:
            log_file = open(_LOG_PATH, "a")
            self._web_process = subprocess.Popen(
                [sys.executable, str(_WEB_PY_PATH)],
                stdout=log_file,
                stderr=log_file,
            )
        logger.info("Webviewer started (pid=%s)", self._web_process.pid)
        self._content_mode = "webviewer"
        self.content_panel.show_webviewer()

    def _update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)

        if self._content_mode == "votes":
            majority_votes = self.orchestrator.get_majority_votes()
            self.content_panel.show_votes(majority_votes)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run(mouse=False)
    app._stop_webviewer()
