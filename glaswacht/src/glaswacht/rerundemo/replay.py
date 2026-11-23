"""Layer A: replay an archive run (results.csv + scans) into Rerun. CPU only."""

import io
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageFilter

from glaswacht.rerundemo import policy, sink
from glaswacht.rerundemo.blueprint import build_blueprint
from glaswacht.rerundemo.data import (
    CLASSES,
    DEFAULT_MODELS,
    Row,
    Stats,
    compute_stats,
    load_results,
)
from glaswacht.rerundemo.style import RATING_COLOURS, colour, md, title

logger = logging.getLogger(__name__)

TIMELINE = "image_idx"
REVIEW_TIMELINE = "review_rank"


def thumbnail(
    path: Path, size: int, quality: int, blur: bool
) -> tuple[bytes, int, int]:
    """JPEG thumbnail; `draft` makes decoding multi-megapixel scans cheap."""
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB")
    im.thumbnail((size, size))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(max(im.size) / 25))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue(), im.width, im.height


def _thumbs(
    rows: list[Row],
    images: Path,
    size: int,
    quality: int,
    blur_flagged: bool,
    workers: int,
) -> Iterator[tuple[bytes, int, int] | None]:
    def one(row: Row):
        try:
            return thumbnail(
                images / row.image, size, quality, blur_flagged and row.flagged
            )
        except Exception as exc:
            logger.warning("no thumbnail for %s: %s", row.image, exc)
            return None

    with ThreadPoolExecutor(workers) as pool:
        yield from pool.map(one, rows)


def rationale_md(model: str, rating: str, text: str) -> str:
    head = f"### {title(model)}: **{rating}**\n\n"
    if rating == "Unknown":
        return (
            head
            + "⚠ **The output could not be parsed.** Raw model output:\n\n```\n"
            + text[:1500]
            + "\n```"
        )
    return head + (text or "*(no rationale)*")


def info_md(row: Row, models: tuple[str, ...], rank: int | None = None) -> str:
    lines = [f"### `{row.image}`", f"**Majority vote: {row.majority}**", ""]
    lines += ["| model | rating |", "|---|---|"]
    lines += [f"| {title(m)} | {row.ratings[m]} |" for m in models]
    uniq = set(row.ratings.values())
    lines += [
        "",
        "✅ unanimous"
        if len(uniq) == 1
        else f"⚠ split verdict ({len(uniq)} different ratings)",
    ]
    if rank is not None:
        lines.append(f"\nReview queue position: **{rank + 1}**")
    return "\n".join(lines)


def log_plate(
    row: Row, thumb: tuple[bytes, int, int] | None, models: tuple[str, ...], rank=None
):
    """Everything about one plate; the caller sets the time before calling."""
    if thumb:
        data, w, h = thumb
        rr.log("plate/image", rr.EncodedImage(contents=data, media_type="image/jpeg"))
        rr.log(
            "plate/verdict",
            rr.Boxes2D(
                mins=[[0, 0]],
                sizes=[[w, h]],
                colors=[RATING_COLOURS[row.majority]],
                labels=[row.majority],
                radii=[-6.0],
            ),
        )
    rr.log("plate/info", md(info_md(row, models, rank)))
    for m in models:
        rr.log(
            f"models/{m}/rationale",
            md(rationale_md(m, row.ratings[m], row.rationales[m])),
        )


def log_stats(stats: Stats) -> None:
    """Per-image time series for the whole archive, sent as columns (fast)."""
    idx = [rr.TimeColumn(TIMELINE, sequence=np.arange(stats.n))]

    def series(path: str, values, name: str, rgb, points: bool = False):
        style = (
            rr.SeriesPoints(colors=[rgb], names=name, marker_sizes=2.0)
            if points
            else rr.SeriesLines(colors=[rgb], names=name, widths=2.0)
        )
        rr.log(path, style, static=True)
        rr.send_columns(
            path,
            indexes=idx,
            columns=rr.Scalars.columns(scalars=np.asarray(values, float)),
        )

    for m in stats.models:
        series(f"stats/unsafe_rate/{m}", stats.unsafe_rate[m], title(m), colour(m))
        series(f"stats/unknown/{m}", stats.unknown_count[m], title(m), colour(m))
        series(
            f"stats/severity/{m}", stats.severity[m], title(m), colour(m), points=True
        )
    pair_colours = [(90, 90, 90), (200, 80, 80), (80, 150, 80)]
    for (a, b), rgb in zip(stats.agree, pair_colours, strict=False):
        series(f"stats/agree/{a}__{b}", stats.agree[(a, b)], f"{a} / {b}", rgb)
    series("stats/agree/all_three", stats.all_agree, "all models", (20, 20, 20))
    series(
        "stats/flagged_total",
        stats.flagged_total,
        "flagged by vote",
        RATING_COLOURS["Unsafe"],
    )
    series(
        "stats/severity/vote",
        stats.majority,
        "majority vote",
        (20, 20, 20),
        points=True,
    )


def log_static_stats(
    stats: Stats, rows: list[Row], archive_total: int | None = None
) -> None:
    for (a, b), mat in stats.confusion.items():
        rr.log(
            f"stats/confusion/{a}__{b}",
            rr.Tensor(mat, dim_names=(title(a), title(b))),
            static=True,
        )
    for m in stats.models:
        rr.log(
            f"stats/distribution/{m}",
            rr.BarChart(stats.distribution[m], color=colour(m)),
            static=True,
        )
    rr.log("stats/findings", md(findings_md(stats, rows, archive_total)), static=True)
    rr.log("stats/confusion_md", md(confusion_md(stats)), static=True)


def confusion_md(stats: Stats) -> str:
    out = ["## Who disagrees with whom", "Rows: first model. Columns: second model.\n"]
    for (a, b), mat in stats.confusion.items():
        out.append(f"**{title(a)} vs {title(b)}**\n")
        out.append("| | " + " | ".join(c.split()[0] for c in CLASSES) + " |")
        out.append("|---|" + "---|" * len(CLASSES))
        for i, c in enumerate(CLASSES):
            out.append(
                f"| {c.split()[0]} | " + " | ".join(str(v) for v in mat[i]) + " |"
            )
        out.append("")
    out.append(f"Bar charts: {', '.join(CLASSES)} in that order.")
    return "\n".join(out)


def findings_md(stats: Stats, rows: list[Row], archive_total: int | None = None) -> str:
    n = stats.n
    unsafe = CLASSES.index("Unsafe")
    pp = CLASSES.index("Potentially Problematic")
    lines = [f"## Findings from {n:,} assessed plates", ""]
    if archive_total:
        lines.append(
            f"- Coverage: **{n / archive_total:.0%}** of the {archive_total:,}-scan archive "
            f"({archive_total - n:,} still to do)."
        )
    for m in stats.models:
        d = stats.distribution[m]
        lines.append(
            f"- **{title(m)}**: {d[unsafe]:,} Unsafe ({d[unsafe] / n:.1%}), "
            f"{d[pp]:,} Potentially Problematic, {d[CLASSES.index('Unknown')]:,} unparseable."
        )
    lines.append(
        f"- All three models agree on **{stats.all_agree[-1]:.1%}** of plates."
    )
    first = stats.models[0]
    outvoted = int(
        ((stats.severity[first] == unsafe) & (stats.majority != unsafe)).sum()
    )
    lines.append(
        f"- The vote flags **{int((stats.majority >= pp).sum()):,}** plates "
        f"({int((stats.majority == unsafe).sum()):,} Unsafe). "
        f"{title(first)} alone says Unsafe on {int((stats.severity[first] == unsafe).sum()):,}, "
        f"but **{outvoted:,}** of those are outvoted by the other two."
    )
    lines.append(
        f"- {len(stats.review_order):,} plates have a split verdict or a parse failure; "
        f"switch the timeline to `{REVIEW_TIMELINE}` to walk them most-contested first."
    )
    return "\n".join(lines)


def replay(
    results: Path,
    images: Path,
    out: Path | None = None,
    connect: str | None = None,
    models: tuple[str, ...] = DEFAULT_MODELS,
    thumb: int = 320,
    quality: int = 65,
    limit: int | None = None,
    blur_flagged: bool = False,
    workers: int = 16,
    archive_total: int | None = None,
) -> None:
    rows = load_results(results, models)[:limit]
    stats = compute_stats(rows, models)
    logger.info("loaded %d rows; %d in review queue", stats.n, len(stats.review_order))

    blueprint = build_blueprint(models)
    sink.init(out, connect, blueprint)
    if out:
        blueprint.save(sink.APP_ID, Path(out).with_suffix(".rbl"))

    policy.log_static(models)
    log_static_stats(stats, rows, archive_total)
    log_stats(stats)

    rank_of = {i: r for r, i in enumerate(stats.review_order)}
    queued: dict[int, tuple[bytes, int, int] | None] = {}
    for i, (row, th) in enumerate(
        zip(
            rows,
            _thumbs(rows, images, thumb, quality, blur_flagged, workers),
            strict=True,
        )
    ):
        rr.set_time(TIMELINE, sequence=i)
        log_plate(row, th, models, rank_of.get(i))
        if i in rank_of:
            queued[i] = th
        if i % 500 == 0:
            logger.info("logged %d/%d", i, stats.n)

    # second pass: the same plates, indexed by review rank only
    rr.reset_time()
    for i, r in rank_of.items():
        rr.set_time(REVIEW_TIMELINE, sequence=r)
        log_plate(rows[i], queued[i], models, r)
    rr.reset_time()

    rr.disconnect()
