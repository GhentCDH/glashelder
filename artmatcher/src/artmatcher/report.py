"""How matching went over a whole `batch-classify` run, measured against labels.

The run says what the classifier found: how many matches, from which sources and
which CLIP ranks. Labels made in the labeler say how often it was right. They are
not a random sample of the images, though: "Next Match" and "Next Negative" look
for one verdict at a time, so matches and non-matches are labeled at different
rates. The estimates therefore work per verdict (a stratum) and scale each one by
its own number of images, which keeps them right however the labeling went.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from artscraper import Artwork
from common.images import is_url

from .batch import artwork_of
from .web.session import LabelDataset

NEAR = 0.05
"""Confidences this close to the threshold count as near it."""
CURVE_POINTS = 101


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a proportion; sound for small samples and 0 or 1."""
    if total == 0:
        return 0.0, 1.0
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    spread /= 1 + z * z / total
    return max(0.0, centre - spread), min(1.0, centre + spread)


@dataclass
class Rate:
    hits: int
    total: int

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.hits, self.total)


@dataclass
class Group:
    """One row of the table by source."""

    name: str
    matches: int = 0
    non_matches: int = 0
    match_confidences: list[float] = field(default_factory=list)
    precision: Rate = field(default_factory=lambda: Rate(0, 0))
    miss_rate: Rate = field(default_factory=lambda: Rate(0, 0))

    @property
    def mean_confidence(self) -> float | None:
        values = self.match_confidences
        return sum(values) / len(values) if values else None


@dataclass
class Accuracy:
    labels: int
    """Labeled images, after the last label of an image replaced earlier ones."""
    comparable: int
    """Labels about the same pair the run chose, which are the ones used."""
    not_in_results: int
    other_artwork: int
    precision: Rate
    """Among images the run calls a match: how many truly are."""
    miss_rate: Rate
    """Among images the run calls no match: how many truly are one."""
    true_matches: tuple[float, float, float] | None
    """Estimated true matches in the whole run: (low, estimate, high)."""
    recall: tuple[float, float, float] | None
    curve: list[tuple[float, float | None, float | None]]
    """(threshold, precision, recall) estimated from the labels."""


@dataclass
class Report:
    source: str
    threshold: float
    run_threshold: float
    candidates: int
    images: int
    matches: int
    near_threshold: int
    partial: bool
    """The run left out results below --min-confidence: non-matches are missing."""
    confidences: np.ndarray
    by_source: list[Group]
    by_rank: list[tuple[int, int]]
    """(rank, matches), rank 1 being CLIP's best; empty when only rank 1 was tried."""
    accuracy: Accuracy | None

    @property
    def match_rate(self) -> float:
        return self.matches / self.images if self.images else 0.0


def _result_source(result: dict, artworks: dict[str, Artwork] | None) -> str:
    if result.get("artwork"):
        return result["artwork"].get("source") or "unknown"
    found = artwork_of(result.get("artwork_path", ""), artworks)
    return found.source if found is not None else "unknown"


def _same_reference(a: str, b: str) -> bool:
    """The same reference image, also when two machines mount local files apart."""
    if a == b:
        return True
    return bool(a and b) and not is_url(a) and Path(a).name == Path(b).name


def build(
    data: dict,
    labels: list[LabelDataset],
    *,
    threshold: float | None = None,
    artworks: dict[str, Artwork] | None = None,
    source: str = "",
) -> Report:
    """The report on a `batch-classify` JSON, with accuracy when labels are given."""
    from .web.source import _current_keys

    metadata = data.get("metadata", {})
    results = [_current_keys(r) for r in data.get("results", [])]
    run_threshold = float(metadata.get("threshold", 0.5))
    threshold = run_threshold if threshold is None else threshold

    confidences = np.array([r["confidence"] for r in results], dtype=float)
    predicted = confidences > threshold
    sources = [_result_source(r, artworks) for r in results]

    groups: dict[str, Group] = {}
    for result, name, is_match in zip(results, sources, predicted, strict=True):
        group = groups.setdefault(name, Group(name))
        if is_match:
            group.matches += 1
            group.match_confidences.append(result["confidence"])
        else:
            group.non_matches += 1

    ranks: dict[int, int] = {}
    for result, is_match in zip(results, predicted, strict=True):
        if is_match:
            rank = int(result.get("rank", 0)) + 1
            ranks[rank] = ranks.get(rank, 0) + 1
    candidates = int(metadata.get("candidates", 1))
    by_rank = sorted(ranks.items()) if candidates > 1 or set(ranks) - {1} else []

    accuracy = (
        _accuracy(results, sources, predicted, confidences, labels, groups)
        if labels
        else None
    )

    return Report(
        source=source,
        threshold=threshold,
        run_threshold=run_threshold,
        candidates=candidates,
        images=len(results),
        matches=int(predicted.sum()),
        near_threshold=int((np.abs(confidences - threshold) <= NEAR).sum()),
        partial=metadata.get("min_confidence") is not None,
        confidences=confidences,
        by_source=sorted(groups.values(), key=lambda g: (-g.matches, g.name)),
        by_rank=by_rank,
        accuracy=accuracy,
    )


def _accuracy(
    results: list[dict],
    sources: list[str],
    predicted: np.ndarray,
    confidences: np.ndarray,
    labels: list[LabelDataset],
    groups: dict[str, Group],
) -> Accuracy:
    # what each label says about the pair it was made for; a later label of
    # the same image (another session, a relabel) replaces an earlier one
    by_index = {r["glashelder_index"]: i for i, r in enumerate(results)}
    by_name = {Path(r["glashelder_path"]).name: i for i, r in enumerate(results)}
    labeled: dict[int | str, tuple[dict, bool]] = {
        entry.get("glashelder_index", entry.get("glashelder_path")): (entry, truth)
        for dataset in labels
        for entry, truth in dataset.labeled()
    }

    comparable: list[tuple[int, bool]] = []
    not_in_results = other_artwork = 0
    for key, (entry, truth) in labeled.items():
        row = by_index.get(key) if isinstance(key, int) else None
        if row is None:
            row = by_name.get(Path(entry.get("glashelder_path", "")).name)
        if row is None:
            not_in_results += 1
        elif not _same_reference(
            entry.get("artwork_path", ""), results[row]["artwork_path"]
        ):
            other_artwork += 1
        else:
            comparable.append((row, truth))

    def rates(rows: list[tuple[int, bool]]) -> tuple[Rate, Rate]:
        matched = [truth for row, truth in rows if predicted[row]]
        missed = [truth for row, truth in rows if not predicted[row]]
        return Rate(sum(matched), len(matched)), Rate(sum(missed), len(missed))

    precision, miss_rate = rates(comparable)
    for name, group in groups.items():
        group.precision, group.miss_rate = rates(
            [(row, truth) for row, truth in comparable if sources[row] == name]
        )

    matches, others = int(predicted.sum()), int((~predicted).sum())
    true_matches = recall = None
    if precision.total and miss_rate.total:
        p, (p_low, p_high) = precision.hits / precision.total, precision.interval
        q, (q_low, q_high) = miss_rate.hits / miss_rate.total, miss_rate.interval
        estimate = matches * p + others * q
        true_matches = (
            matches * p_low + others * q_low,
            estimate,
            matches * p_high + others * q_high,
        )

        def share(found: float, missing: float) -> float:
            return found / (found + missing) if found + missing else 0.0

        recall = (
            share(matches * p_low, others * q_high),
            share(matches * p, others * q),
            share(matches * p_high, others * q_low),
        )

    return Accuracy(
        labels=len(labeled),
        comparable=len(comparable),
        not_in_results=not_in_results,
        other_artwork=other_artwork,
        precision=precision,
        miss_rate=miss_rate,
        true_matches=true_matches,
        recall=recall,
        curve=_curve(comparable, predicted, confidences, precision, miss_rate),
    )


def _curve(
    comparable: list[tuple[int, bool]],
    predicted: np.ndarray,
    confidences: np.ndarray,
    precision: Rate,
    miss_rate: Rate,
) -> list[tuple[float, float | None, float | None]]:
    """Precision and recall at other thresholds, from the labels alone.

    Each label stands for (images with its verdict) / (labels with that verdict)
    images, which undoes labeling the two verdicts at different rates.
    """
    if not (precision.total and miss_rate.total):
        return []
    weight = {
        True: predicted.sum() / precision.total,
        False: (~predicted).sum() / miss_rate.total,
    }
    rows = np.array([row for row, _ in comparable])
    truth = np.array([t for _, t in comparable])
    weights = np.array([weight[bool(predicted[row])] for row, _ in comparable])
    scores = confidences[rows]

    curve = []
    for t in np.linspace(0, 1, CURVE_POINTS):
        called = scores > t
        found = weights[called & truth].sum()
        wrong = weights[called & ~truth].sum()
        missed = weights[~called & truth].sum()
        curve.append(
            (
                float(t),
                float(found / (found + wrong)) if found + wrong else None,
                float(found / (found + missed)) if found + missed else None,
            )
        )
    return curve


def charts(report: Report) -> dict[str, str]:
    """The report's charts, as base64 PNGs for embedding in the HTML."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plot import ACCENT, ACCENT2, ACCENT3, ACCENT4, setup_style

    setup_style()
    found: dict[str, str] = {}

    def keep(name: str, fig) -> None:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        plt.close(fig)
        found[name] = base64.b64encode(buffer.getvalue()).decode()

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.hist(report.confidences, bins=100, range=(0, 1), color=ACCENT, edgecolor="none")
    ax.axvline(report.threshold, color=ACCENT4, lw=1.5, label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("confidence")
    ax.set_ylabel("images (log)")
    ax.legend()
    keep("confidence", fig)

    if report.by_source:
        names = [g.name for g in report.by_source][::-1]
        fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(names)))
        ax.barh(names, [g.matches for g in report.by_source][::-1], color=ACCENT2)
        ax.set_xlabel("matches")
        keep("sources", fig)

    if report.by_rank:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            [f"#{r}" for r, _ in report.by_rank],
            [n for _, n in report.by_rank],
            color=ACCENT3,
        )
        ax.set_xlabel("CLIP rank of the match")
        ax.set_ylabel("matches")
        keep("ranks", fig)

    if report.accuracy and report.accuracy.curve:
        curve = report.accuracy.curve
        fig, ax = plt.subplots(figsize=(9, 3.2))
        xs = [t for t, _, _ in curve]
        ax.plot(
            xs,
            [p if p is not None else np.nan for _, p, _ in curve],
            color=ACCENT,
            label="precision",
        )
        ax.plot(
            xs,
            [r if r is not None else np.nan for _, _, r in curve],
            color=ACCENT2,
            label="recall",
        )
        ax.axvline(report.threshold, color=ACCENT4, lw=1, ls=":", label="threshold")
        ax.set_xlabel("threshold")
        ax.set_ylim(0, 1.02)
        ax.legend()
        keep("curve", fig)

    return found


def render(report: Report) -> str:
    """The report as one self-contained HTML page."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(),
    )
    return environment.get_template("report.html").render(
        report=report, charts=charts(report)
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def summary_lines(report: Report) -> list[str]:
    """The few lines worth seeing in a terminal; the HTML has the rest."""
    lines = [
        f"{report.images:,} images, threshold {report.threshold:g}"
        + (
            f" (run at {report.run_threshold:g})"
            if report.threshold != report.run_threshold
            else ""
        ),
        f"matches  {report.matches:,} ({_percent(report.match_rate)}), "
        f"{report.near_threshold:,} within ±{NEAR} of the threshold",
        "  "
        + "   ".join(f"{g.name} {g.matches:,}" for g in report.by_source if g.matches),
    ]
    if report.by_rank:
        lines.append(
            "  by CLIP rank: " + ", ".join(f"#{r} {n:,}" for r, n in report.by_rank)
        )
    if report.accuracy is not None:
        a = report.accuracy
        lines.append(
            f"labels   {a.labels:,} ({a.comparable:,} usable), "
            f"precision {_percent(a.precision.value)} of {a.precision.total}, "
            f"miss rate {_percent(a.miss_rate.value)} of {a.miss_rate.total}"
        )
        if a.true_matches and a.recall:
            low, mid, high = a.true_matches
            lines.append(
                f"estimated true matches ~{mid:,.0f} ({low:,.0f} to {high:,.0f}), "
                f"recall ~{_percent(a.recall[1])} "
                f"({_percent(a.recall[0])} to {_percent(a.recall[2])})"
            )
        else:
            lines.append("  (label both a match and a non-match to estimate recall)")
    return lines
