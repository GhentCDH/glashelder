"""The 25-plate demo: the probe traces, animated, plus just enough context to talk around."""

import logging
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from glaswacht.rerundemo import policy, sink
from glaswacht.rerundemo.data import Row, compute_stats, load_results
from glaswacht.rerundemo.deepdive import load_traces
from glaswacht.rerundemo.generation import (
    BG,
    CANVAS_W,
    TICK,
    canvas_height,
    log_generation,
    plate_index_md,
    scoreboard_size,
    topk_size,
)
from glaswacht.rerundemo.replay import findings_md
from glaswacht.rerundemo.style import md, title

logger = logging.getLogger(__name__)

APP_ID = "glaswacht-demo"
WINDOW = 260  # ticks of history the curves show, roughly one plate


def _recent():
    return rrb.VisibleTimeRange(
        TICK,
        start=rrb.TimeRangeBoundary.cursor_relative(seq=-WINDOW),
        end=rrb.TimeRangeBoundary.cursor_relative(seq=0),
    )


def _fit(w: int, h: int) -> rrb.VisualBounds2D:
    """Pin a view to one fixed-size image: newer viewers otherwise leave wide margins."""
    return rrb.VisualBounds2D(x_range=[0, w], y_range=[0, h])


def demo_blueprint(
    models: tuple[str, ...], canvas_h: int, archive: bool
) -> rrb.Blueprint:
    def canvas(m: str) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(
            origin=f"gen/{m}/canvas",
            name=title(m),
            background=BG,
            visual_bounds=rrb.VisualBounds2D(
                x_range=[-8, CANVAS_W + 8], y_range=[-56, canvas_h]
            ),
        )

    def topk(m: str) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(
            origin=f"gen/{m}/topk",
            name="next token",
            background=BG,
            visual_bounds=_fit(*topk_size()),
        )

    generation = rrb.Vertical(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="deep/plate", name="Plate", background=BG),
                rrb.TextDocumentView(origin="deep/plate/info", name="About"),
                row_shares=[1, 1],
            ),
            *[rrb.Vertical(canvas(m), topk(m), row_shares=[3, 2]) for m in models],
            column_shares=[2.2] + [3] * len(models),
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="deep/scoreboard",
                name="Decisive token",
                background=BG,
                visual_bounds=_fit(*scoreboard_size(len(models))),
            ),
            rrb.TimeSeriesView(
                origin="curves/confidence",
                name="Confidence",
                axis_y=rrb.ScalarAxis(range=(0.0, 1.05)),
                time_ranges=_recent(),
                plot_legend=rrb.PlotLegend(visible=False),
            ),
            rrb.TimeSeriesView(
                origin="curves/surprise",
                name="Surprise (bits)",
                time_ranges=_recent(),
                plot_legend=rrb.PlotLegend(visible=False),
            ),
            column_shares=[3, 2, 2],
        ),
        row_shares=[5, 2],
        name="Token by token",
    )
    verdicts = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(origin="deep/plate", name="Plate", background=BG),
            rrb.Spatial2DView(
                origin="deep/scoreboard",
                name="Decisive token",
                background=BG,
                visual_bounds=_fit(*scoreboard_size(len(models))),
            ),
            row_shares=[3, 1],
        ),
        *[
            rrb.TextDocumentView(origin=f"deep/verdict/{m}", name=title(m))
            for m in models
        ],
        column_shares=[3] + [2] * len(models),
        name="Verdicts",
    )
    sees = rrb.Vertical(
        rrb.Horizontal(
            *[
                rrb.Spatial2DView(
                    origin=f"deep/seen/{m}", name=f"{title(m)} sees", background=BG
                )
                for m in models
            ]
        ),
        rrb.Horizontal(
            *[
                rrb.TextDocumentView(origin=f"deep/seen/{m}/stats", name=title(m))
                for m in models
            ]
        ),
        row_shares=[4, 1],
        name="What the models see",
    )
    context = rrb.Horizontal(
        rrb.TextDocumentView(origin="policy/pipeline", name="Pipeline"),
        rrb.Vertical(
            rrb.TextDocumentView(origin="policy/prompt", name="The prompt"),
            rrb.TextDocumentView(origin="policy/models", name="Models"),
        ),
        rrb.TextDocumentView(origin="policy/taxonomy", name="Policy (taxonomy.txt)"),
        name="Policy and prompt",
    )
    plates = rrb.Horizontal(
        rrb.TextDocumentView(origin="deep/index", name="Plate index"),
        *(
            [
                rrb.TextDocumentView(
                    origin="stats/findings", name="Findings, whole batch"
                )
            ]
            if archive
            else []
        ),
        name="Plates and findings",
    )
    cost = rrb.Vertical(
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="ops/latency", name="Seconds per plate"),
            rrb.TimeSeriesView(origin="ops/tokens", name="Tokens generated"),
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="ops/image_tokens", name="Image tokens in the prompt"
            ),
            rrb.TimeSeriesView(origin="ops/vram", name="VRAM allocated (GB)"),
            rrb.TextLogView(origin="ops/lifecycle", name="Lifecycle"),
        ),
        name="Cost",
    )
    return rrb.Blueprint(
        rrb.Tabs(generation, verdicts, sees, context, plates, cost, active_tab=0),
        rrb.TimePanel(timeline=TICK, fps=12.0, state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
        collapse_panels=False,
    )


def build_demo(
    probe_dir: Path,
    results: Path | None = None,
    out: Path | None = None,
    connect: str | None = None,
) -> None:
    run = load_traces(probe_dir)
    models = run.models
    rows: list[Row] = load_results(results, models) if results else []
    archive = {r.image: r for r in rows}

    blueprint = demo_blueprint(models, canvas_height(run), bool(rows))
    sink.init(out, connect, blueprint, app_id=APP_ID)
    if out:
        blueprint.save(APP_ID, Path(out).with_suffix(".rbl"))

    policy.log_static(models)
    if rows:
        rr.log(
            "stats/findings",
            md(findings_md(compute_stats(rows, models), rows)),
            static=True,
        )
    starts = log_generation(run, archive)
    rr.log("deep/index", md(plate_index_md(run, starts, archive)), static=True)
    logger.info(
        "logged %d plates; timeline %s, plate starts %s",
        len(run.plates),
        TICK,
        starts[:5],
    )
    rr.disconnect()
