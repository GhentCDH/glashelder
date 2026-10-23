import logging
import os
import secrets
import subprocess
import sys
import threading
from io import TextIOWrapper
from pathlib import Path
from urllib.parse import urlencode

from textual.app import App
from textual.containers import Horizontal, VerticalScroll
from textual.markup import escape
from textual.reactive import reactive
from textual.widgets import Static

from glaswacht.models.utils import RESULTS_CSV, WEB_PORT, get_prompt
from glaswacht.orchestrator import ModelStatus, Orchestrator, Skip

_LOG_PATH = Path("glaswacht.log")
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
    compact: reactive[bool] = reactive(False, layout=True)

    def watch_compact(self, compact: bool):
        self.set_class(compact, "compact")

    def render(self):
        if self.compact:
            return (
                f"[bold {_SEPIA_BRIGHT}]GLASWACHT[/bold {_SEPIA_BRIGHT}]  "
                f"[{_SEPIA_MID}]needs an RTX 4090 (24GB VRAM) or better[/{_SEPIA_MID}]"
            )
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
            f"[dim]{escape(state.current_task)}[/dim]"
            if state.current_task
            else f"[{_SEPIA_MID}]—[/{_SEPIA_MID}]"
        )
        skipped_display = (
            f"[{_SEPIA_DANGER}]{state.skipped}[/{_SEPIA_DANGER}]"
            if state.skipped
            else f"[{_SEPIA_MID}]0[/{_SEPIA_MID}]"
        )

        self.update(
            f"[{_SEPIA_MID}]STATUS :[/{_SEPIA_MID}] {_status_markup(state.status)}\n"
            f"[{_SEPIA_MID}]TASK   :[/{_SEPIA_MID}] {task_display}\n"
            f"[{_SEPIA_MID}]IMAGE  :[/{_SEPIA_MID}] {image_display}\n"
            f"[{_SEPIA_MID}]RATING :[/{_SEPIA_MID}] {rating_markup}\n"
            f"[{_SEPIA_MID}]SKIPPED:[/{_SEPIA_MID}] {skipped_display}"
        )


class ContentPanel(Static):
    """Scrollable panel that shows either majority votes or the taxonomy."""

    def show_votes(self, majority_votes: dict, skips: list[Skip]):
        # skips first, the vote list can be hundreds of lines long
        self.update(self._skips_markup(skips) + self._votes_markup(majority_votes))

    @staticmethod
    def _skips_markup(skips: list[Skip]) -> str:
        if not skips:
            return ""
        header = (
            f"[bold {_SEPIA_DANGER}]\\[ SKIPPED ][/bold {_SEPIA_DANGER}]"
            f" [{_SEPIA_MID}]({len(skips)} skipped, details in {_LOG_PATH})[/{_SEPIA_MID}]\n"
            f"[{_SEPIA_BORDER}]{'─' * 72}[/{_SEPIA_BORDER}]\n"
        )
        skip_lines = [
            f"[{_SEPIA_MID}]{escape(skip.image):<32}[/{_SEPIA_MID}]  "
            f"[{_SEPIA_BRIGHT}]{skip.model}[/{_SEPIA_BRIGHT}]  "
            f"[{_SEPIA_DANGER}]{escape(skip.error)}[/{_SEPIA_DANGER}]"
            for skip in skips
        ]
        return header + "\n".join(skip_lines) + "\n\n"

    @staticmethod
    def _votes_markup(majority_votes: dict) -> str:
        if not majority_votes:
            return (
                f"[bold {_SEPIA_BRIGHT}]\\[ MAJORITY VOTES ][/bold {_SEPIA_BRIGHT}]\n\n"
                f"[{_SEPIA_MID}]Awaiting results...[/{_SEPIA_MID}]"
            )

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
            f"[bold {_SEPIA_BRIGHT}]\\[ MAJORITY VOTES ][/bold {_SEPIA_BRIGHT}]"
            f" [{_SEPIA_MID}]({len(majority_votes)} images processed)[/{_SEPIA_MID}]\n"
            f"{'  |  '.join(summary_parts)}\n"
            f"[{_SEPIA_BORDER}]{'─' * 72}[/{_SEPIA_BORDER}]\n"
        )
        vote_lines = [
            f"[{_SEPIA_MID}]{name:<32}[/{_SEPIA_MID}]  {_rating_markup(vote)}"
            for name, vote in majority_votes.items()
        ]
        return header + "\n".join(vote_lines)

    def show_taxonomy(self):
        try:
            text = get_prompt(str(TAXONOMY_PATH))
        except Exception as e:
            text = f"Could not load taxonomy: {e}"
        self.update(
            f"[bold {_SEPIA_BRIGHT}]\\[ TAXONOMY ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]{text}[/{_SEPIA_MID}]"
        )

    def show_webviewer(self, url: str):
        self.update(
            f"[bold {_SEPIA_BRIGHT}]\\[ VALIDATE USING WEBVIEWER ][/bold {_SEPIA_BRIGHT}]\n\n"
            f"[{_SEPIA_MID}]The validation web server is running. "
            f"Open in your browser:[/{_SEPIA_MID}]\n\n"
            f"[bold {_SEPIA_BRIGHT}]{url}[/bold {_SEPIA_BRIGHT}]"
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
            self.update(f"[dim {_SEPIA_MID}]{self._label}[/dim {_SEPIA_MID}]")
        elif self.selected:
            self.update(
                f"[bold {_SEPIA_BG} on {_SEPIA_BRIGHT}]"
                f"> {self._label} <"
                f"[/bold {_SEPIA_BG} on {_SEPIA_BRIGHT}]"
            )
        else:
            self.update(f"[{_SEPIA_BRIGHT}]{self._label}[/{_SEPIA_BRIGHT}]")


_MENU_LABELS = ["RUN", "TAXONOMY", "VALIDATE", "QUIT"]

# the full logo needs this much room, else the header is one line
_LOGO_MIN_WIDTH = 88
_LOGO_MIN_HEIGHT = 40

_SCROLL_HINT = " ←/→ menu · enter select · ↑/↓ pgup/pgdn scroll "


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

    #header.compact {{
        padding: 0 2;
    }}

    #models {{
        height: auto;
        margin: 0 1;
    }}

    ModelPanel {{
        border: double {_SEPIA_BORDER};
        border-title-color: {_SEPIA_BRIGHT};
        border-title-style: bold;
        padding: 0 2;
        height: auto;
        width: 1fr;
        margin: 0 0 0 1;
        color: {_SEPIA_MID};
    }}

    ModelPanel:first-of-type, MenuItem:first-of-type {{
        margin: 0;
    }}

    #content-scroll {{
        border: double {_SEPIA_BORDER};
        border-subtitle-color: {_SEPIA_MID};
        height: 1fr;
        margin: 0 1;
        padding: 0 1;
        color: {_SEPIA_MID};
    }}

    #menu {{
        height: auto;
        margin: 0 1;
    }}

    MenuItem {{
        width: 1fr;
        height: 3;
        border: solid {_SEPIA_BORDER};
        content-align: center middle;
        margin: 0 0 0 1;
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
        self._web_log_fh: TextIOWrapper | None = None
        self._web_secret = secrets.token_urlsafe(32)

    def compose(self):
        yield GlaswachtHeader(id="header")
        yield Horizontal(id="models")
        # the app scrolls it from on_key, so the menu keeps the arrow keys
        yield VerticalScroll(id="content-scroll", can_focus=False)
        yield Horizontal(id="menu")

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
        self.content_panel.show_votes({}, [])
        content_container.mount(self.content_panel)
        content_container.border_subtitle = _SCROLL_HINT

        for i, label in enumerate(_MENU_LABELS):
            item = MenuItem(label)
            item.selected = i == 0
            self.menu_widgets.append(item)
            menu_container.mount(item)

        self._fit_header()
        self.set_interval(0.1, self._update_display)

    def on_resize(self):
        self._fit_header()

    def _fit_header(self):
        header = self.query_one(GlaswachtHeader)
        header.compact = (
            self.size.width < _LOGO_MIN_WIDTH or self.size.height < _LOGO_MIN_HEIGHT
        )

    def on_key(self, event):
        content = self.query_one("#content-scroll", VerticalScroll)
        scroll = {
            "up": content.scroll_up,
            "down": content.scroll_down,
            "pageup": content.scroll_page_up,
            "pagedown": content.scroll_page_down,
            "home": content.scroll_home,
            "end": content.scroll_end,
        }
        key = event.key
        if key == "left":
            self._move_selection(-1)
        elif key == "right":
            self._move_selection(1)
        elif key in scroll:
            scroll[key](animate=False)
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
            self._web_process.wait()
        self._web_process = None
        if self._web_log_fh:
            self._web_log_fh.close()
            self._web_log_fh = None

    def _activate(self):
        label = _MENU_LABELS[self._selected_index]
        if label != "VALIDATE":
            self._stop_webviewer()
        if label == "RUN":
            self._start_run()
        elif label == "TAXONOMY":
            self._show_taxonomy()
        elif label == "VALIDATE":
            self._start_webviewer()
        elif label == "QUIT":
            self.exit()

    def _start_run(self):
        if self._is_running:
            return
        logger.info("RUN requested, path=%s", self._image_path)
        self._is_running = True
        self._set_mode("votes")
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
        self.menu_widgets[0].enabled = True
        self.menu_widgets[1].enabled = True
        self.menu_widgets[2].enabled = True

    def _set_mode(self, mode: str):
        """Show `mode` ("votes", "taxonomy" or "webviewer") in the content panel."""
        if mode != self._content_mode:
            self.query_one("#content-scroll", VerticalScroll).scroll_home(animate=False)
        self._content_mode = mode

    def _show_taxonomy(self):
        self._set_mode("taxonomy")
        if self.content_panel:
            self.content_panel.show_taxonomy()

    def _start_webviewer(self):
        if self._image_path is None:
            self.notify(
                "No --path provided. Use: glaswacht tui --path <dir>",
                severity="error",
            )
            return
        images_dir = Path(self._image_path)
        if images_dir.is_file():
            images_dir = images_dir.parent

        # Only launch if not already running
        if self._web_process is None or self._web_process.poll() is not None:
            self._web_log_fh = open(_LOG_PATH, "a")
            self._web_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "glaswacht",
                    "serve",
                    "--no-open",
                    "--images-dir",
                    str(images_dir),
                    "--csv-path",
                    str(RESULTS_CSV),
                    "--port",
                    str(WEB_PORT),
                ],
                stdout=self._web_log_fh,
                stderr=self._web_log_fh,
                env={**os.environ, "GLASHELDER_SECRET": self._web_secret},
            )
        logger.info("Webviewer started (pid=%s)", self._web_process.pid)
        self._set_mode("webviewer")
        if self.content_panel:
            query = urlencode({"secret": self._web_secret})
            self.content_panel.show_webviewer(f"http://localhost:{WEB_PORT}/?{query}")

    def _update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            self.panels[model_id].update_state(model_state)

        if self._content_mode == "votes":
            majority_votes = self.orchestrator.get_majority_votes()
            if self.content_panel:
                self.content_panel.show_votes(
                    majority_votes, self.orchestrator.get_skips()
                )
