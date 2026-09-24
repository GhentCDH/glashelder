"""Standalone diagnostics TUI: trace per-token generation for a single image
across one or more models, then hand off to the `diagnostics_web` viewer.

Unlike the general monitor (`tui.py`), everything here — which image, which
models, top_k, max_new_tokens, out_dir, port — is entered and can be changed
interactively inside the tool instead of only at launch, since a diagnostics
pass only ever concerns one image at a time.
"""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from textual.app import App
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static

from glaswacht.models.safety_model import SafetyModel
from glaswacht.orchestrator import ModelStatus, Orchestrator
from glaswacht.tui_common import (
    DIAGNOSTICS as PALETTE,
)
from glaswacht.tui_common import (
    LOG_PATH,
    LOGO_DIAG,
    VIEWER_STARTUP_GRACE,
    GlaswachtHeader,
    MenuItem,
    ModelPanel,
    base_css,
    elapsed,
    logger,
    progress_bar,
    start_viewer,
    stop_viewer,
)

DIAGNOSTICS_PY_PATH = Path(__file__).parent / "diagnostics_web.py"


@dataclass
class Row:
    key: str
    kind: str  # "action" | "toggle" | "edit"
    render: Callable[[], str]
    on_activate: Callable[[], None] | None = None
    get_value: Callable[[], str] | None = None
    apply: Callable[[str], "str | None"] | None = None


class DiagContentPanel(Static):
    """Scrollable panel showing the settings summary, run progress, or result."""

    def show_idle(self, summary: str):
        p = PALETTE
        self.update(
            f"[bold {p.bright}][ DIAGNOSE ][/bold {p.bright}]\n\n"
            f"[{p.mid}]{summary}[/{p.mid}]\n\n"
            f"[{p.mid}]Press RUN DIAGNOSTICS to start.[/{p.mid}]"
        )

    def show_progress(self, image_name: str, status: str, progress: str = ""):
        p = PALETTE
        progress_block = f"\n\n{progress}" if progress else ""
        self.update(
            f"[bold {p.bright}][ DIAGNOSE ][/bold {p.bright}]\n\n"
            f"[{p.mid}]IMAGE  :[/{p.mid}] [dim]{image_name}[/dim]\n"
            f"[{p.mid}]STATUS :[/{p.mid}] [{p.bright}]{status}[/{p.bright}]"
            f"{progress_block}"
        )

    def show_result(self, image_name: str, status: str, url: str = ""):
        p = PALETTE
        url_block = (
            f"\n\n[{p.mid}]Open in your browser:[/{p.mid}] " f"[bold {p.bright}]{url}[/bold {p.bright}]"
            if url
            else ""
        )
        self.update(
            f"[bold {p.bright}][ DIAGNOSE ][/bold {p.bright}]\n\n"
            f"[{p.mid}]IMAGE  :[/{p.mid}] [dim]{image_name}[/dim]\n"
            f"[{p.mid}]STATUS :[/{p.mid}] [{p.bright}]{status}[/{p.bright}]"
            f"{url_block}"
        )


class DiagnosticsApp(App):
    # Textual auto-focuses the first focusable widget (the shared Input) on
    # mount unless this is disabled, which would silently start the app in
    # "editing" mode before the user has pressed anything.
    AUTO_FOCUS = None

    CSS = base_css(
        palette=PALETTE,
        extra=f"""
    #menu {{ height: 2fr; }}

    #edit-bar {{
        height: auto;
        margin: 0 1 1 1;
        padding: 1 2;
        border: solid {PALETTE.border};
        display: none;
    }}

    #edit-bar.open {{ display: block; }}

    #edit-error {{ color: {PALETTE.danger}; }}
    """,
    )

    def __init__(
        self,
        model_registry: dict[str, type[SafetyModel]],
        initial_model_names: tuple[str, ...],
        initial_top_k: int,
        initial_max_new_tokens: int | None,
        initial_out_dir: str,
        initial_port: int,
    ):
        super().__init__()
        self._model_registry = model_registry
        self._selected_models: dict[str, bool] = {
            name: name in initial_model_names for name in model_registry
        }
        self._image_path = ""
        self._top_k = initial_top_k
        self._max_new_tokens = initial_max_new_tokens
        self._out_dir = initial_out_dir
        self._port = initial_port
        self._custom_position = 0
        self._custom_text = ""

        self.orchestrator: Orchestrator = self._build_orchestrator()
        self.panels: dict[str, ModelPanel] = {}
        self.content_panel: DiagContentPanel | None = None
        self._rows: list[Row] = self._build_rows()
        self.menu_widgets: list[MenuItem] = []
        self._selected_index = 0

        self._mode = "menu"  # "menu" | "editing"
        self._editing_key: str | None = None

        self._is_running = False
        self._content_mode = "idle"  # "idle" | "running" | "result"
        self._viewers: dict[str, tuple] = {}
        self._run_started = 0.0
        self._run_image_name = ""
        self._run_summary = ""

    # ------------------------------------------------------------------ #
    # Rows: labels, values, validators
    # ------------------------------------------------------------------ #

    def _build_rows(self) -> list[Row]:
        rows = [
            Row(
                "image",
                "edit",
                self._row_image_label,
                get_value=lambda: self._image_path,
                apply=self._apply_image,
            )
        ]
        for name in self._model_registry:
            rows.append(
                Row(
                    f"model:{name}",
                    "toggle",
                    (lambda n=name: self._row_model_label(n)),
                    on_activate=(lambda n=name: self._toggle_model(n)),
                )
            )
        rows += [
            Row(
                "top_k",
                "edit",
                self._row_top_k_label,
                get_value=lambda: str(self._top_k),
                apply=self._apply_top_k,
            ),
            Row(
                "max_new_tokens",
                "edit",
                self._row_max_new_tokens_label,
                get_value=self._row_max_new_tokens_value,
                apply=self._apply_max_new_tokens,
            ),
            Row(
                "out_dir",
                "edit",
                self._row_out_dir_label,
                get_value=lambda: self._out_dir,
                apply=self._apply_out_dir,
            ),
            Row(
                "port",
                "edit",
                self._row_port_label,
                get_value=lambda: str(self._port),
                apply=self._apply_port,
            ),
            Row(
                "custom_position",
                "edit",
                self._row_custom_position_label,
                get_value=lambda: str(self._custom_position),
                apply=self._apply_custom_position,
            ),
            Row(
                "custom_text",
                "edit",
                self._row_custom_text_label,
                get_value=lambda: self._custom_text,
                apply=self._apply_custom_text,
            ),
            Row("run", "action", lambda: "RUN DIAGNOSTICS", on_activate=self._start_run),
            Row("quit", "action", lambda: "QUIT", on_activate=self.exit),
        ]
        return rows

    def _row_image_label(self) -> str:
        return f"IMAGE: {self._image_path or '(not set)'}"

    def _row_model_label(self, name: str) -> str:
        mark = "x" if self._selected_models[name] else " "
        return f"[{mark}] {name}"

    def _row_top_k_label(self) -> str:
        return f"TOP_K: {self._top_k}"

    def _row_max_new_tokens_value(self) -> str:
        return "" if self._max_new_tokens is None else str(self._max_new_tokens)

    def _row_max_new_tokens_label(self) -> str:
        value = "(model default)" if self._max_new_tokens is None else str(self._max_new_tokens)
        return f"MAX_NEW_TOKENS: {value}"

    def _row_out_dir_label(self) -> str:
        return f"OUT_DIR: {self._out_dir}"

    def _row_port_label(self) -> str:
        return f"PORT: {self._port}"

    def _row_custom_position_label(self) -> str:
        return f"CUSTOM_POSITION: {self._custom_position}"

    def _row_custom_text_label(self) -> str:
        text = self._custom_text or "(none — no extra counterfactual)"
        return f"CUSTOM_TEXT: {text}"

    def _apply_image(self, raw: str) -> str | None:
        path = raw.strip()
        if not path:
            return "Image path cannot be empty."
        if not os.path.isfile(path):
            return f"Not a file: {path}"
        self._image_path = path
        return None

    def _apply_top_k(self, raw: str) -> str | None:
        try:
            value = int(raw.strip())
        except ValueError:
            return "TOP_K must be an integer."
        if value <= 0:
            return "TOP_K must be positive."
        self._top_k = value
        return None

    def _apply_max_new_tokens(self, raw: str) -> str | None:
        raw = raw.strip()
        if not raw:
            self._max_new_tokens = None
            return None
        try:
            value = int(raw)
        except ValueError:
            return "MAX_NEW_TOKENS must be an integer, or blank for the model default."
        if value <= 0:
            return "MAX_NEW_TOKENS must be positive."
        self._max_new_tokens = value
        return None

    def _apply_out_dir(self, raw: str) -> str | None:
        out_dir = raw.strip()
        if not out_dir:
            return "OUT_DIR cannot be empty."
        self._out_dir = out_dir
        return None

    def _apply_port(self, raw: str) -> str | None:
        try:
            value = int(raw.strip())
        except ValueError:
            return "PORT must be an integer."
        if not (1 <= value <= 65535):
            return "PORT must be between 1 and 65535."
        self._port = value
        return None

    def _apply_custom_position(self, raw: str) -> str | None:
        try:
            value = int(raw.strip())
        except ValueError:
            return "CUSTOM_POSITION must be an integer."
        if value < 0:
            return "CUSTOM_POSITION must be zero or positive."
        self._custom_position = value
        return None

    def _apply_custom_text(self, raw: str) -> str | None:
        # Blank is valid: it means "no extra counterfactual this run", the
        # default. Anything else is forced verbatim, so no further validation
        # here — a per-model tokenizer decides at run time whether it encodes
        # to any tokens at all (see SafetyModel._custom_counterfactual).
        self._custom_text = raw.strip()
        return None

    # ------------------------------------------------------------------ #
    # Model selection <-> Orchestrator/panels
    # ------------------------------------------------------------------ #

    def _build_orchestrator(self) -> Orchestrator:
        classes = [
            cls for name, cls in self._model_registry.items() if self._selected_models[name]
        ]
        return Orchestrator(classes)

    def _rebuild_model_panels(self):
        container = self.query_one("#models")
        container.remove_children()
        self.panels.clear()
        for model_id, state in self.orchestrator.get_state().items():
            panel = ModelPanel(model_name=state.name, palette=PALETTE)
            panel.update_state(state)
            self.panels[model_id] = panel
            container.mount(panel)

    def _toggle_model(self, name: str):
        if self._selected_models[name] and sum(self._selected_models.values()) == 1:
            self.notify("At least one model must stay selected.", severity="warning")
            return
        self._selected_models[name] = not self._selected_models[name]
        self.orchestrator = self._build_orchestrator()
        self._rebuild_model_panels()
        self._refresh_row(f"model:{name}")
        self._refresh_idle_summary()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def compose(self):
        yield GlaswachtHeader(id="header", palette=PALETTE, logo=LOGO_DIAG)
        yield Horizontal(id="models")
        yield VerticalScroll(id="content-scroll")
        yield VerticalScroll(id="menu")
        yield Vertical(
            Static("", id="edit-label"),
            Input(id="edit-input"),
            Static("", id="edit-error"),
            id="edit-bar",
        )

    def on_mount(self):
        logger.info("Diagnostics TUI started")
        content_container = self.query_one("#content-scroll")
        menu_container = self.query_one("#menu")

        self._rebuild_model_panels()

        self.content_panel = DiagContentPanel()
        self.content_panel.show_idle(self._settings_summary())
        content_container.mount(self.content_panel)

        for i, row in enumerate(self._rows):
            item = MenuItem(row.render(), palette=PALETTE)
            item.selected = i == 0
            self.menu_widgets.append(item)
            menu_container.mount(item)

        self.set_interval(0.1, self._update_display)

    def _refresh_idle_summary(self):
        if self._content_mode == "idle" and self.content_panel:
            self.content_panel.show_idle(self._settings_summary())

    def _settings_summary(self) -> str:
        models = ", ".join(name for name, on in self._selected_models.items() if on)
        custom = (
            f"   CUSTOM: @{self._custom_position} {self._custom_text!r}"
            if self._custom_text
            else ""
        )
        return (
            f"MODELS: {models or '(none)'}   TOP_K: {self._top_k}   "
            f"MAX_NEW_TOKENS: {self._max_new_tokens or 'default'}   "
            f"OUT_DIR: {self._out_dir}   PORT: {self._port}{custom}"
        )

    # ------------------------------------------------------------------ #
    # Menu navigation / edit-mode state machine
    # ------------------------------------------------------------------ #

    def on_key(self, event):
        key = event.key
        if self._mode == "editing":
            if key == "escape":
                event.stop()
                self._exit_edit(refresh=False)
            # Swallow everything else here (notably up/down, which Input does
            # not bind and would otherwise leak into _move_selection); Input
            # itself handles printable characters and enter (-> submit).
            return

        if key == "up":
            event.stop()
            self._move_selection(-1)
        elif key == "down":
            event.stop()
            self._move_selection(1)
        elif key in ("enter", "return"):
            # Stop this event *before* possibly focusing the shared Input
            # inside _activate_selected_row (edit rows): if we don't, this
            # same "enter" keypress goes on to be re-delivered to the newly
            # focused Input, whose own enter-binding immediately submits the
            # value we just prefilled, instantly exiting edit mode again.
            event.stop()
            self._activate_selected_row()

    def _move_selection(self, delta: int):
        n = len(self._rows)
        new = (self._selected_index + delta) % n
        for _ in range(n):
            if self.menu_widgets[new].enabled:
                self.menu_widgets[self._selected_index].selected = False
                self._selected_index = new
                self.menu_widgets[new].selected = True
                self.query_one("#menu").scroll_to_widget(self.menu_widgets[new], animate=False)
                return
            new = (new + delta) % n

    def _select_row(self, key: str):
        idx = next(i for i, r in enumerate(self._rows) if r.key == key)
        self.menu_widgets[self._selected_index].selected = False
        self._selected_index = idx
        self.menu_widgets[idx].selected = True

    def _refresh_row(self, key: str):
        idx = next(i for i, r in enumerate(self._rows) if r.key == key)
        self.menu_widgets[idx].set_label(self._rows[idx].render())

    def _set_rows_enabled(self, enabled: bool):
        """Enable/disable every row except QUIT, which always stays usable."""
        for item, row in zip(self.menu_widgets, self._rows, strict=True):
            if row.key != "quit":
                item.enabled = enabled

    def _activate_selected_row(self):
        row = self._rows[self._selected_index]
        if row.kind == "edit":
            self._start_edit(row)
        else:
            row.on_activate()

    def _start_edit(self, row: Row):
        self._mode = "editing"
        self._editing_key = row.key
        self.query_one("#edit-label", Static).update(
            f"[{PALETTE.bright}]Editing {row.key.upper()}[/{PALETTE.bright}]  "
            f"[dim]Enter=confirm, Esc=cancel[/dim]"
        )
        self.query_one("#edit-error", Static).update("")
        input_widget = self.query_one("#edit-input", Input)
        input_widget.value = row.get_value() if row.get_value else ""
        self.query_one("#edit-bar").add_class("open")
        # Deferred: Textual's App._on_key (the framework's own private key
        # handler) always runs right after this public on_key handler
        # returns, for the *same* keypress, and re-checks bindings against
        # whatever is focused *at that moment*. Focusing the Input
        # synchronously here would make that immediate re-check see Input's
        # own enter->submit binding and fire it instantly, using the just-set
        # prefill value as if the user had submitted it. Waiting a refresh
        # lets this keypress's dispatch finish first.
        self.call_after_refresh(self.set_focus, input_widget)

    def _exit_edit(self, refresh: bool):
        key = self._editing_key
        self._mode = "menu"
        self._editing_key = None
        self.query_one("#edit-bar").remove_class("open")
        self.set_focus(None)
        if refresh and key:
            self._refresh_row(key)
            self._refresh_idle_summary()

    def on_input_submitted(self, event: Input.Submitted):
        if self._mode != "editing" or self._editing_key is None:
            return
        row = next(r for r in self._rows if r.key == self._editing_key)
        error = row.apply(event.value) if row.apply else None
        if error:
            self.query_one("#edit-error", Static).update(f"[{PALETTE.danger}]{error}[/{PALETTE.danger}]")
            return
        self._exit_edit(refresh=True)

    # ------------------------------------------------------------------ #
    # RUN flow
    # ------------------------------------------------------------------ #

    def stop_viewers(self):
        stop_viewer(self._viewers)

    def _start_run(self):
        if self._is_running:
            return
        if not self._image_path:
            self.notify("Set IMAGE first.", severity="error")
            return
        if not os.path.isfile(self._image_path):
            self.notify(f"Image not found: {self._image_path}", severity="error")
            return
        if not any(self._selected_models.values()):
            self.notify("Select at least one model.", severity="error")
            return

        image = self._image_path
        out_dir = self._out_dir
        top_k = self._top_k
        max_new_tokens = self._max_new_tokens
        port = self._port
        orchestrator = self.orchestrator
        custom_counterfactual = (
            (self._custom_position, self._custom_text) if self._custom_text else None
        )

        logger.info("Diagnostics RUN requested, image=%s", image)
        self._is_running = True
        self._content_mode = "running"
        self._run_image_name = os.path.basename(image)
        self._run_started = time.monotonic()
        self._set_rows_enabled(False)
        self._select_row("quit")
        if self.content_panel:
            self.content_panel.show_progress(
                self._run_image_name, "Starting diagnostics across selected models..."
            )

        def _run():
            try:
                traces = orchestrator.run_diagnostics(
                    image,
                    out_dir,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    custom_counterfactual=custom_counterfactual,
                )
            except Exception as exc:
                logger.error("Diagnostics failed", exc_info=True)
                self.call_from_thread(
                    self._on_run_complete, image, out_dir, port, [], f"Failed: {exc}"
                )
                return
            self.call_from_thread(self._on_run_complete, image, out_dir, port, traces, "")

        threading.Thread(target=_run, daemon=True).start()

    def _on_run_complete(
        self, image: str, out_dir: str, port: int, traces: list, error: str
    ):
        logger.info("Diagnostics complete (%d traces)", len(traces))
        self._is_running = False
        self._set_rows_enabled(True)
        name = os.path.basename(image)

        if error or not traces:
            self._content_mode = "result"
            if self.content_panel:
                self.content_panel.show_result(name, error or "No traces were written.")
            return

        process = start_viewer(
            self._viewers,
            "diagnostics",
            DIAGNOSTICS_PY_PATH,
            env={
                "GW_DIAG_DIR": str(Path(out_dir).resolve()),
                "GW_DIAG_PORT": str(port),
            },
        )
        logger.info("Diagnostics viewer started (pid=%s)", process.pid)
        self._content_mode = "result"
        self._run_summary = f"{len(traces)} trace(s) written to {out_dir}/"
        if self.content_panel:
            self.content_panel.show_result(name, f"{self._run_summary} — starting viewer...")
        self.set_timer(VIEWER_STARTUP_GRACE, lambda: self._check_diagnostics_viewer(name, port))

    def _check_diagnostics_viewer(self, name: str, port: int):
        viewer = self._viewers.get("diagnostics")
        if viewer is None or self._content_mode != "result":
            return
        process, _ = viewer
        if process.poll() is None:
            if self.content_panel:
                self.content_panel.show_result(
                    name, self._run_summary, url=f"http://localhost:{port}"
                )
            return

        code = process.returncode
        stop_viewer(self._viewers, "diagnostics")
        logger.error("Diagnostics viewer exited with code %s", code)
        if self.content_panel:
            self.content_panel.show_result(
                name,
                f"{self._run_summary}\nViewer exited immediately (code {code}). "
                f"See {LOG_PATH} for the traceback.",
            )

    def _render_run_progress(self):
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
        elapsed_str = elapsed(time.monotonic() - self._run_started)

        if current is None:
            status = f"Writing traces... ({finished}/{len(state)} models)"
            bar, detail = progress_bar(0, 0, palette=PALETTE), elapsed_str
        else:
            status = (
                f"{current.name} — {current.phase or 'working'} ({finished + 1}/{len(state)})"
            )
            bar = progress_bar(current.progress_done, current.progress_total, palette=PALETTE)
            detail = (
                f"{current.progress_done}/{current.progress_total} tokens   {elapsed_str}"
                if current.progress_total
                else elapsed_str
            )

        self.content_panel.show_progress(
            self._run_image_name,
            status,
            progress=f"{bar}  [{PALETTE.mid}]{detail}[/{PALETTE.mid}]",
        )

    def _update_display(self):
        state = self.orchestrator.get_state()
        for model_id, model_state in state.items():
            if model_id in self.panels:
                self.panels[model_id].update_state(model_state)

        if self._content_mode == "running" and self._is_running:
            self._render_run_progress()


if __name__ == "__main__":
    from glaswacht.cli import MODEL_REGISTRY

    app = DiagnosticsApp(
        model_registry=MODEL_REGISTRY,
        initial_model_names=tuple(MODEL_REGISTRY.keys()),
        initial_top_k=5,
        initial_max_new_tokens=None,
        initial_out_dir="diagnostics",
        initial_port=5002,
    )
    app.run(mouse=False)
    app.stop_viewers()
