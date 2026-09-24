"""Widgets, palette and helpers shared by the Glaswacht TUIs.

`tui.py` (RUN / TAXONOMY / VALIDATE) and `diagnostics_tui.py` (DIAGNOSE) are
separate Textual apps with separate menus and separate flows, but they share
the same look: the sepia header, the per-model status panels, the
keyboard-driven menu list, and the subprocess plumbing for spawning a Flask
viewer alongside the TUI.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

from textual.reactive import reactive
from textual.widgets import Static

LOG_PATH = Path(__file__).parent / "glaswacht.log"
logger = logging.getLogger("glaswacht")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_PATH)
    _fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    )
    logger.addHandler(_fh)

LOGO = """\
░██████╗░██╗░░░░░░█████╗░░██████╗░██╗░░░░░░░██╗░█████╗░░█████╗░██╗░░██╗████████╗
██╔════╝░██║░░░░░██╔══██╗██╔════╝░██║░░██╗░░██║██╔══██╗██╔══██╗██║░░██║╚══██╔══╝
██║░░██╗░██║░░░░░███████║╚█████╗░░╚██╗████╗██╔╝███████║██║░░╚═╝███████║░░░██║░░░
██║░░╚██╗██║░░░░░██╔══██║░╚═══██╗░░████╔═████║░██╔══██║██║░░██╗██╔══██║░░░██║░░░
╚██████╔╝███████╗██║░░██║██████╔╝░░╚██╔╝░╚██╔╝░██║░░██║╚█████╔╝██║░░██║░░░██║░░░
░╚═════╝░╚══════╝╚═╝░░╚═╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░"""

LOGO_DIAG = """\
░██████╗░░██╗░░░░░░░██╗░░░░░░██████╗░██╗░█████╗░░██████╗░███╗░░██╗░█████╗░░██████╗████████╗██╗░█████╗░
██╔════╝░░██║░░██╗░░██║░░░░░░██╔══██╗██║██╔══██╗██╔════╝░████╗░██║██╔══██╗██╔════╝╚══██╔══╝██║██╔══██╗
██║░░██╗░░╚██╗████╗██╔╝█████╗██║░░██║██║███████║██║░░██╗░██╔██╗██║██║░░██║╚█████╗░░░░██║░░░██║██║░░╚═╝
██║░░╚██╗░░████╔═████║░╚════╝██║░░██║██║██╔══██║██║░░╚██╗██║╚████║██║░░██║░╚═══██╗░░░██║░░░██║██║░░██╗
╚██████╔╝░░╚██╔╝░╚██╔╝░░░░░░░██████╔╝██║██║░░██║╚██████╔╝██║░╚███║╚█████╔╝██████╔╝░░░██║░░░██║╚█████╔╝
░╚═════╝░░░░╚═╝░░░╚═╝░░░░░░░░╚═════╝░╚═╝╚═╝░░╚═╝░╚═════╝░╚═╝░░╚══╝░╚════╝░╚═════╝░░░░╚═╝░░░╚═╝░╚════╝░"""


INFO = "Scripted by @bvercru and @mielpeeters at The Ghent Centre for Digital Humanities (https://www.ghentcdh.ugent.be/).\nRequirements for Glaswacht: Geforce RTX 4090 (24GB VRAM) or better. "


@dataclass(frozen=True)
class Palette:
    bright: str  # primary text
    mid: str  # labels / dim text
    safe: str
    warn: str
    danger: str
    border: str
    bg: str


# Sepia palette — the general monitor TUI (tui.py: RUN/TAXONOMY/VALIDATE).
SEPIA_BRIGHT = "#F5DEB3"  # wheat  - primary text
SEPIA_MID = "#C8A96E"  # tan    - labels / dim text
SEPIA_SAFE = "#D4A843"  # amber  - safe
SEPIA_WARN = "#C87941"  # burnt sienna - potentially problematic
SEPIA_DANGER = "#8B2500"  # dark rust - unsafe / error
SEPIA_BORDER = "#8B6914"  # dark gold - borders
SEPIA_BG = "#1A1008"  # near-black brown - background

SEPIA = Palette(
    bright=SEPIA_BRIGHT,
    mid=SEPIA_MID,
    safe=SEPIA_SAFE,
    warn=SEPIA_WARN,
    danger=SEPIA_DANGER,
    border=SEPIA_BORDER,
    bg=SEPIA_BG,
)

# Cyan/teal "lab console" palette — the diagnostics TUI (diagnostics_tui.py),
# deliberately cool where the monitor is warm so the two tools read as
# different instruments at a glance.
DIAGNOSTICS = Palette(
    bright="#E8F6FF",  # icy cyan-white - primary text
    mid="#7FA8B8",  # steel cyan - labels / dim text
    safe="#2FD9A0",  # teal-green - safe
    warn="#E8B339",  # amber - potentially problematic
    danger="#FF5D5D",  # coral red - unsafe / error
    border="#1E5F74",  # deep teal - borders
    bg="#0B1420",  # near-black graphite-navy - background
)

# A viewer that dies on startup (missing results file, port already taken) exits
# almost immediately, so re-check shortly after spawning rather than reporting
# success just because Popen returned.
VIEWER_STARTUP_GRACE = 1.5


def rating_markup(rating: str, palette: Palette = SEPIA) -> str:
    if rating == "Safe":
        return f"[{palette.safe}]{rating}[/{palette.safe}]"
    elif rating == "Potentially Problematic":
        return f"[{palette.warn}]{rating}[/{palette.warn}]"
    elif rating:
        return f"[{palette.danger}]{rating}[/{palette.danger}]"
    return f"[dim {palette.mid}]—[/dim {palette.mid}]"


def progress_bar(
    done: int, total: int, width: int = 34, palette: Palette = SEPIA
) -> str:
    """A block-character bar. With no known total the phase is still running but
    unmeasurable (a model load), so render an empty trough rather than a fake bar."""
    if total <= 0:
        return f"[{palette.border}]{'░' * width}[/{palette.border}]"
    fraction = min(max(done / total, 0.0), 1.0)
    filled = int(fraction * width)
    return (
        f"[{palette.safe}]{'█' * filled}[/{palette.safe}]"
        f"[{palette.border}]{'░' * (width - filled)}[/{palette.border}]"
    )


def elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def status_markup(status, palette: Palette = SEPIA) -> str:
    from glaswacht.orchestrator import ModelStatus

    color_map = {
        ModelStatus.CHILLING: palette.mid,
        ModelStatus.LOADING: palette.warn,
        ModelStatus.RUNNING: palette.bright,
        ModelStatus.DONE: palette.safe,
        ModelStatus.ERROR: palette.danger,
    }
    color = color_map.get(status, palette.mid)
    return f"[{color}]{status.value}[/{color}]"


class GlaswachtHeader(Static):
    def __init__(self, palette: Palette = SEPIA, logo: str = LOGO, **kwargs):
        super().__init__(**kwargs)
        self._palette = palette
        self._logo = logo

    def render(self):
        p = self._palette
        return f"[{p.bright}]{self._logo}[/{p.bright}]\n[{p.mid}]{INFO}[/{p.mid}]"


class ModelPanel(Static):
    def __init__(self, model_name: str, palette: Palette = SEPIA, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._palette = palette

    def on_mount(self):
        self.border_title = f" {self._model_name} "

    def update_state(self, state):
        p = self._palette
        rating = rating_markup(state.latest_rating, p)
        image_display = (
            f"[dim]{state.latest_image}[/dim]"
            if state.latest_image
            else f"[{p.mid}]—[/{p.mid}]"
        )
        task_display = (
            f"[dim]{state.current_task}[/dim]"
            if state.current_task
            else f"[{p.mid}]—[/{p.mid}]"
        )

        self.update(
            f"[{p.mid}]STATUS :[/{p.mid}] {status_markup(state.status, p)}\n"
            f"[{p.mid}]TASK   :[/{p.mid}] {task_display}\n"
            f"[{p.mid}]IMAGE  :[/{p.mid}] {image_display}\n"
            f"[{p.mid}]RATING :[/{p.mid}] {rating}"
        )


class MenuItem(Static):
    selected: reactive[bool] = reactive(False)
    enabled: reactive[bool] = reactive(True)

    def __init__(self, label: str, palette: Palette = SEPIA, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._palette = palette

    def on_mount(self):
        self._refresh_label()

    def set_label(self, label: str):
        self._label = label
        self._refresh_label()

    def watch_selected(self, _):
        self._refresh_label()

    def watch_enabled(self, _):
        self._refresh_label()

    def _refresh_label(self):
        p = self._palette
        if not self.enabled:
            self.update(f"[dim {p.mid}]  {self._label}[/dim {p.mid}]")
        elif self.selected:
            self.update(
                f"[bold {p.bg} on {p.bright}]  > {self._label} <[/bold {p.bg} on {p.bright}]"
            )
        else:
            self.update(f"[{p.bright}]  {self._label}[/{p.bright}]")


def base_css(palette: Palette = SEPIA, extra: str = "") -> str:
    """Screen/header/model-panel/menu rules shared by every Glaswacht TUI."""
    p = palette
    return f"""
    Screen {{
        background: {p.bg};
        color: {p.mid};
        layout: vertical;
    }}

    #header {{
        border: double {p.border};
        padding: 1 2;
        height: auto;
        margin: 1 1 0 1;
    }}

    #models {{
        height: auto;
        margin: 0 0;
    }}

    ModelPanel {{
        border: double {p.border};
        border-title-color: {p.bright};
        border-title-style: bold;
        padding: 1 2;
        height: auto;
        width: 1fr;
        margin: 0 1;
        color: {p.mid};
    }}

    #content-scroll {{
        border: double {p.border};
        height: 1fr;
        margin: 0 1;
        padding: 0 2;
        color: {p.mid};
    }}

    #menu {{
        height: auto;
        margin: 0 1 0 1;
    }}

    MenuItem {{
        width: 100%;
        height: 3;
        padding: 0 2;
        border: solid {p.border};
        margin: 0;
    }}

    {extra}
    """


def start_viewer(
    viewers: dict[str, tuple[subprocess.Popen, TextIOWrapper]],
    key: str,
    script: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn `script` as a subprocess, tracked under `key` in `viewers` so a
    second call reuses the still-running process instead of doubling up."""
    running = viewers.get(key)
    if running is not None:
        if running[0].poll() is None:
            return running[0]
        stop_viewer(viewers, key)
    log_fh = open(LOG_PATH, "a")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=log_fh,
        stderr=log_fh,
        env={**os.environ, **env} if env else None,
    )
    viewers[key] = (process, log_fh)
    return process


def stop_viewer(
    viewers: dict[str, tuple[subprocess.Popen, TextIOWrapper]], key: str | None = None
):
    for name in list(viewers) if key is None else [key]:
        viewer = viewers.pop(name, None)
        if viewer is None:
            continue
        process, log_fh = viewer
        if process.poll() is None:
            process.terminate()
            process.wait()
        log_fh.close()
