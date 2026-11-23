"""The viewer layout: one tab per part of the story, in the order you would tell it."""

import rerun.blueprint as rrb

from glaswacht.rerundemo.data import DEFAULT_MODELS
from glaswacht.rerundemo.style import title


def overview(models) -> rrb.Container:
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(origin="plate", name="Plate + vote"),
            rrb.TextDocumentView(origin="plate/info", name="Verdicts"),
            row_shares=[3, 1],
        ),
        _rationales_column(models),
        column_shares=[2, 3],
        name="1 · Plate by plate",
    )


def _rationales_column(models):
    return rrb.Vertical(
        *[rrb.TextDocumentView(origin=f"models/{m}", name=title(m)) for m in models]
    )


def archive(models) -> rrb.Container:
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.TextDocumentView(origin="stats/findings", name="Findings"),
            rrb.Grid(
                *[
                    rrb.BarChartView(
                        origin=f"stats/distribution/{m}", name=f"{title(m)} ratings"
                    )
                    for m in models
                ]
            ),
            column_shares=[2, 3],
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="stats/unsafe_rate", name="Running Unsafe-rate per model"
            ),
            rrb.TimeSeriesView(origin="stats/agree", name="Running agreement"),
            rrb.TimeSeriesView(
                origin="stats/flagged_total", name="Plates flagged by the vote"
            ),
        ),
        rrb.Horizontal(
            *[
                rrb.TensorView(
                    origin=f"stats/confusion/{a}__{b}", name=f"{title(a)} vs {title(b)}"
                )
                for a, b in pairs
            ],
            rrb.TextDocumentView(origin="stats/confusion_md", name="Confusion tables"),
        ),
        row_shares=[3, 3, 3],
        name="2 · The archive",
    )


def review(models) -> rrb.Container:
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(origin="plate", name="Plate + vote"),
            rrb.TextDocumentView(origin="plate/info", name="Verdicts"),
            row_shares=[3, 1],
        ),
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="stats/severity", name="Rating per plate (0 safe … 3 unknown)"
            ),
            _rationales_column(models),
            row_shares=[1, 3],
        ),
        column_shares=[2, 3],
        name="3 · Review queue",
    )


def policy(models) -> rrb.Container:
    return rrb.Grid(
        rrb.TextDocumentView(origin="policy/pipeline", name="Pipeline"),
        rrb.TextDocumentView(origin="policy/taxonomy", name="Policy (taxonomy.txt)"),
        rrb.Vertical(
            rrb.TextDocumentView(origin="policy/models", name="Models"),
            rrb.TextDocumentView(origin="policy/prompt", name="The prompt"),
        ),
        grid_columns=3,
        name="4 · Policy and prompt",
    )


def build_blueprint(models=DEFAULT_MODELS, timeline: str = "image_idx"):
    tabs = [overview(models), archive(models), review(models), policy(models)]
    return rrb.Blueprint(
        rrb.Tabs(*tabs, active_tab=0),
        rrb.TimePanel(timeline=timeline),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )
