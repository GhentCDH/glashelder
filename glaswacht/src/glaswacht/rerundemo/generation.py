"""Token-by-token generation, rendered for Rerun.

Each generated token becomes a small image tile, coloured by the probability the model
gave it, placed where it lands in the text. Rerun shows everything logged up to the time
cursor, so scrubbing or playing the `deep_tick` timeline makes the three models "type"
their answers side by side. Next to each one, a labelled top-k chart shows the
alternatives the model weighed at that step.

Markdown in Rerun cannot colour text, hence the rendered tiles.
"""

import io
import itertools
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageDraw, ImageFont

from glaswacht.models.safety_model import TokenStep, Trace
from glaswacht.models.utils import calculate_majority_vote
from glaswacht.rerundemo.data import Row, normalise, pattern
from glaswacht.rerundemo.deepdive import DeepRun, Plate, mosaic, patch_grid
from glaswacht.rerundemo.style import RATING_COLOURS, colour, md, title

TICK = "deep_tick"
PLATE = "plate"
PADDING = 12  # quiet ticks after the slowest model finishes, before the next plate

# canvas geometry, in canvas units (1 unit = 1 screen px at 100% zoom)
CANVAS_W = 680
FONT_PX = 15
LINE_H = 24
TILE_H = 21
SCALE = 2  # tiles are rendered at 2x and scaled down, so text stays crisp when zoomed

BG = (22, 24, 29)
TEXT = (201, 209, 217)
DIM = (110, 118, 129)
TOP_K = 6

_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
)


@cache
def font(px: int, mono: bool = True, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold.ttf" if bold else ".ttf"
    for d in _FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), px)
    return ImageFont.load_default(px)


# --------------------------------------------------------------------------- colour

# probability of the chosen token -> tile background, green (sure) to red (torn).
# The green end is kept dark so the bulk of near-certain text stays calm and the eye
# goes to the yellow-to-red places where the model hesitated.
_STOPS = [
    (1.00, (30, 84, 50)),
    (0.90, (46, 125, 62)),
    (0.75, (120, 170, 50)),
    (0.55, (222, 196, 40)),
    (0.35, (236, 130, 36)),
    (0.00, (214, 44, 44)),
]


def prob_colour(p: float) -> tuple[int, int, int]:
    p = max(0.0, min(1.0, p))
    for (p_hi, c_hi), (p_lo, c_lo) in itertools.pairwise(_STOPS):
        if p >= p_lo:
            t = (p_hi - p) / (p_hi - p_lo)
            return tuple(int(a + (b - a) * t) for a, b in zip(c_hi, c_lo, strict=True))
    return _STOPS[-1][1]


def text_on(bg: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if bg is None:
        return TEXT
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (20, 20, 24) if lum > 120 else (238, 240, 243)


# --------------------------------------------------------------------------- layout


@dataclass
class Seg:
    step: int  # -1 for the pre-filled scaffold
    text: str
    x: float
    y: float
    w: float


def char_w() -> float:
    return font(FONT_PX).getlength("M")


def layout(trace: Trace, width: int = CANVAS_W) -> list[Seg]:
    """Place every token like a text editor would.

    Tiles abut, so sub-word tokens read as one word. When a word does not fit, the
    whole word (all its tokens so far) moves to the next line, and a leading space
    hangs off the left edge so the new line starts flush.
    """
    cw = char_w()
    pieces = [(-1, trace.prefill)] + [(s.index, s.token) for s in trace.steps]
    x = y = 0.0
    segs: list[Seg] = []
    word_from = 0  # index in segs of the first piece of the current word
    for step, text in pieces:
        for j, part in enumerate(text.split("\n")):
            if j:
                x, y = 0.0, y + LINE_H
                word_from = len(segs)
            if not part:
                continue
            starts_word = part[0].isspace() or x == 0
            if starts_word:
                word_from = len(segs)
            w = cw * len(part)
            if x > 0 and x + w > width:
                head = segs[word_from:] if not starts_word else []
                if head and head[0].x > 0:  # carry the started word along
                    dx = head[0].x - (cw if head[0].text[0].isspace() else 0)
                    for h in head:
                        h.x -= dx
                        h.y += LINE_H
                    x, y = x - dx, y + LINE_H
                else:
                    x, y = (-cw if part[0].isspace() else 0.0), y + LINE_H
            segs.append(Seg(step, part, x, y, w))
            x += w
    return segs


def canvas_height(run: DeepRun) -> int:
    h = max(
        max((s.y for s in layout(t)), default=0)
        for p in run.plates
        for t in p.traces.values()
    )
    return int(h + LINE_H * 2)


# --------------------------------------------------------------------------- rendering


# Rerun (0.43+) plays a series of encoded images on one entity as a video stream, and
# refuses frames whose format changes: same size, same bit depth, every time. So every
# image that changes over time is padded to a fixed size and encoded the same way.


def png(im: Image.Image, colours: int = 48) -> bytes:
    """Palette PNG, always 8-bit: the palette is padded to 256 entries so a frame with
    few colours does not become a 4-bit PNG."""
    q = im.quantize(colours, method=Image.Quantize.FASTOCTREE)
    pal = q.getpalette()[: 3 * 256]
    q.putpalette(pal + [0] * (3 * 256 - len(pal)))
    buf = io.BytesIO()
    q.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def jpeg(
    pixels: np.ndarray | Image.Image, size: tuple[int, int], centre: bool
) -> tuple[bytes, int, int]:
    """JPEG on a fixed-size canvas. Returns the bytes and where the picture landed."""
    im = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
    im = im.convert("RGB")
    canvas = Image.new("RGB", size, BG)
    ox = (size[0] - im.width) // 2 if centre else 0
    oy = (size[1] - im.height) // 2 if centre else 0
    canvas.paste(im, (ox, oy))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=88, subsampling=2)
    return buf.getvalue(), ox, oy


def render_tile(text: str, bg, fg) -> np.ndarray:
    """One token, as raw RGB pixels. The highlight sits behind the visible characters
    only, inset by a pixel so neighbouring highlighted tokens stay distinguishable.

    Raw rather than PNG on purpose: newer viewers play a series of encoded images on
    one entity as a video stream, which must keep one size, and tokens differ in width.
    """
    cw = char_w() * SCALE
    w = max(1, round(cw * len(text)))
    h = TILE_H * SCALE
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    if bg is not None and text.strip():
        d.rounded_rectangle(
            (lead * cw + SCALE, SCALE, w - trail * cw - SCALE, h - SCALE),
            radius=3 * SCALE,
            fill=bg,
        )
    d.text((0, h / 2), text, font=font(FONT_PX * SCALE), fill=fg, anchor="lm")
    return np.asarray(im)


def shown(token: str) -> str:
    """Make whitespace visible in the alternatives list."""
    vis = token.replace("\n", "⏎").replace("\t", "→").replace(" ", "·")
    return vis if vis else "∅"


def topk_size() -> tuple[int, int]:
    """Pixel size of a top-k chart (always TOP_K rows: generate keeps 8 candidates)."""
    return 360 * SCALE, (44 + 26 * TOP_K + 10) * SCALE


def scoreboard_size(n_models: int) -> tuple[int, int]:
    return 900 * SCALE, (52 + 44 * n_models + 30) * SCALE


def render_topk(m: str, step: TokenStep, n_steps: int) -> bytes:
    """Labelled bar chart of the alternatives at one step; the chosen one in model colour."""
    s = SCALE
    w, row, head = 360 * s, 26 * s, 44 * s
    rows = step.top[:TOP_K]
    im = Image.new("RGB", topk_size(), BG)
    d = ImageDraw.Draw(im)
    d.text(
        (12 * s, 8 * s),
        f"step {step.index + 1}/{n_steps}",
        font=font(13 * s, mono=False, bold=True),
        fill=TEXT,
    )
    d.text(
        (w - 12 * s, 8 * s),
        f"entropy {step.entropy:.2f}",
        font=font(12 * s, mono=False),
        fill=DIM,
        anchor="ra",
    )
    d.text(
        (12 * s, 26 * s),
        "what could come next",
        font=font(11 * s, mono=False),
        fill=DIM,
    )
    bar_x0, bar_x1 = 128 * s, w - 64 * s
    for k, (tok, p) in enumerate(rows):
        y = head + k * row
        chosen = tok == step.token or (k == 0 and abs(p - step.prob) < 1e-9)
        fg = TEXT if chosen else DIM
        label = shown(tok)
        if len(label) > 11:
            label = label[:10] + "…"
        d.text((12 * s, y + row / 2), label, font=font(13 * s), fill=fg, anchor="lm")
        d.rounded_rectangle(
            (bar_x0, y + 6 * s, bar_x1, y + row - 6 * s),
            radius=3 * s,
            fill=(36, 40, 48),
        )
        bw = max(2 * s, (bar_x1 - bar_x0) * p)
        bar_c = colour(m) if chosen else (88, 96, 110)
        d.rounded_rectangle(
            (bar_x0, y + 6 * s, bar_x0 + bw, y + row - 6 * s), radius=3 * s, fill=bar_c
        )
        txt = f"{p:.3f}" if p >= 0.001 else "<.001"
        d.text(
            (w - 12 * s, y + row / 2),
            txt,
            font=font(12 * s),
            fill=fg,
            anchor="rm",
        )
    return png(im, 64)


def render_scoreboard(plate: Plate, models, archive: Row | None) -> bytes:
    """Per model: how the first generated token splits over the three ratings."""
    s = SCALE
    labels = ["Safe", "Potentially Problematic", "Unsafe"]
    row, head = 44 * s, 52 * s
    im = Image.new("RGB", scoreboard_size(len(models)), BG)
    d = ImageDraw.Draw(im)
    d.text(
        (16 * s, 10 * s),
        'The decisive token: P(rating) right after the pre-filled  {"rating": "',
        font=font(14 * s, mono=False, bold=True),
        fill=TEXT,
    )
    x0, x1 = 170 * s, 640 * s
    for k, m in enumerate(models):
        t = plate.traces[m]
        y = head + k * row
        d.text(
            (16 * s, y + row / 2),
            title(m),
            font=font(14 * s, mono=False),
            fill=colour(m),
            anchor="lm",
        )
        x = x0
        probs = [t.label_probs.get(lbl, 0.0) for lbl in labels]
        probs.append(max(0.0, 1 - sum(probs)))
        for lbl, p in zip([*labels, "other"], probs, strict=True):
            seg = (x1 - x0) * p
            c = RATING_COLOURS.get(lbl, (70, 74, 84))
            if seg >= 1:
                d.rectangle((x, y + 8 * s, x + seg, y + row - 8 * s), fill=c)
            if seg > 44 * s:
                d.text(
                    (x + seg / 2, y + row / 2),
                    f"{p:.0%}",
                    font=font(13 * s, mono=False, bold=True),
                    fill=text_on(c),
                    anchor="mm",
                )
            x += seg
        probe = normalise(t.rating.rating)
        d.text(
            (660 * s, y + row / 2 - 8 * s),
            "greedy:",
            font=font(11 * s, mono=False),
            fill=DIM,
            anchor="lm",
        )
        d.text(
            (712 * s, y + row / 2 - 8 * s),
            probe,
            font=font(12 * s, mono=False, bold=True),
            fill=RATING_COLOURS[probe],
            anchor="lm",
        )
        if archive is not None:
            arch = archive.ratings[m]
            d.text(
                (660 * s, y + row / 2 + 9 * s),
                "archive:",
                font=font(11 * s, mono=False),
                fill=DIM,
                anchor="lm",
            )
            d.text(
                (720 * s, y + row / 2 + 9 * s),
                arch,
                font=font(12 * s, mono=False),
                fill=RATING_COLOURS[arch],
                anchor="lm",
            )
    ly = head + row * len(models) + 8 * s
    lx = x0
    for lbl in [*labels, "other"]:
        c = RATING_COLOURS.get(lbl, (70, 74, 84))
        d.rectangle((lx, ly + 3 * s, lx + 12 * s, ly + 15 * s), fill=c)
        d.text(
            (lx + 18 * s, ly + 9 * s),
            lbl,
            font=font(11 * s, mono=False),
            fill=DIM,
            anchor="lm",
        )
        lx += (len(lbl) * 7 + 50) * s
    return png(im, 128)


def render_legend() -> bytes:
    s = SCALE
    w, h = 330 * s, 40 * s
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    d.text(
        (10 * s, 4 * s),
        "highlight = how sure the model was of the token it wrote",
        font=font(11 * s, mono=False),
        fill=DIM,
    )
    x0, x1, y0, y1 = 10 * s, w - 10 * s, 20 * s, 30 * s
    for x in range(x0, x1):
        p = 1 - (x - x0) / (x1 - x0)
        d.line((x, y0, x, y1), fill=prob_colour(p) or BG)
    d.rectangle((x0, y0, x1, y1), outline=(60, 66, 78))
    for p in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        x = x0 + (1 - p) * (x1 - x0)
        d.text(
            (x, y1 + 2 * s),
            f"{p:.1f}",
            font=font(9 * s, mono=False),
            fill=DIM,
            anchor="ma",
        )
    return png(im, 64)


# --------------------------------------------------------------------------- logging


# where tiles of the previous plate are parked: far outside the canvas' fixed bounds
OFFSTAGE = [-1e6, -1e6, 0]


def _tile(path: str, seg: Seg, bg, fg) -> None:
    rr.log(
        path,
        rr.Transform3D(translation=[seg.x, seg.y, 0], scale=1 / SCALE),
        rr.Image(render_tile(seg.text, bg, fg), color_model="RGB"),
    )


def _box(seg_list: list[Seg], rgb, width: float = 1.5) -> rr.Boxes2D:
    return rr.Boxes2D(
        mins=[[s.x - 1, s.y - 1] for s in seg_list],
        sizes=[[s.w + 2, TILE_H + 2] for s in seg_list],
        colors=[rgb],
        radii=[-width],
    )


def plate_index_md(run: DeepRun, starts: list[int], archive: dict[str, Row]) -> str:
    rows = [
        "## The 25 plates",
        "Drag the time cursor to a start tick to jump to a plate.\n",
        "| # | start tick | pattern in the archive | archive vote | now (greedy) |",
        "|---|---|---|---|---|",
    ]
    for p, (plate, start) in enumerate(zip(run.plates, starts, strict=True)):
        image = next(iter(plate.traces.values())).image
        a = archive.get(image)
        now = calculate_majority_vote(
            [normalise(t.rating.rating) for t in plate.traces.values()]
        )
        rows.append(
            f"| {p + 1} | {start} | {pattern(a) if a else '?'} | "
            f"{a.majority if a else '?'} | {now} |"
        )
    return "\n".join(rows)


def log_generation(run: DeepRun, archive: dict[str, Row] | None = None) -> list[int]:
    """Log every plate's animated generation. Returns the start tick of each plate."""
    archive = archive or {}
    models = run.models
    legend = render_legend()
    for m in models:
        # the colour key sits above each canvas, right-aligned
        rr.log(
            f"gen/{m}/canvas/legend",
            rr.Transform3D(translation=[CANVAS_W - 330, -48, 0], scale=1 / SCALE),
            rr.EncodedImage(contents=legend, media_type="image/png"),
            static=True,
        )
        rr.log(
            f"curves/confidence/{m}",
            rr.SeriesLines(colors=[colour(m)], names=title(m), widths=2.0),
            static=True,
        )
        rr.log(
            f"curves/surprise/{m}",
            rr.SeriesLines(colors=[colour(m)], names=title(m), widths=2.0),
            static=True,
        )
        for name in ("latency", "vram", "image_tokens", "tokens"):
            rr.log(
                f"ops/{name}/{m}",
                rr.SeriesPoints(colors=[colour(m)], names=title(m), marker_sizes=6.0),
                static=True,
            )

    rr.set_time(TICK, sequence=0)
    for m in models:
        rr.log(
            "ops/lifecycle",
            rr.TextLog(
                f"{title(m)} loaded in {run.load_s.get(m, 0):.1f}s", level="INFO"
            ),
        )

    # fixed frame sizes for everything that changes per plate (see `png`)
    aspects = {}
    for pl in run.plates:
        with Image.open(run.plate_dir / pl.name / "plate.jpg") as im:
            aspects[pl.name] = im.width / im.height
    plate_px = max(
        max(Image.open(run.plate_dir / pl.name / "plate.jpg").size) for pl in run.plates
    )
    seen_size = {
        m: tuple(
            max(dim)
            for dim in zip(
                *(
                    mosaic(pl.traces[m].seen, aspects[pl.name]).shape[1::-1]
                    for pl in run.plates
                    if pl.traces[m].seen
                ),
                strict=True,
            )
        )
        for m in models
        if any(pl.traces[m].seen for pl in run.plates)
    }

    shown_tiles: dict[str, set[str]] = {m: set() for m in models}
    starts = []
    base = 0
    for p, plate in enumerate(run.plates):
        starts.append(base)
        image = next(iter(plate.traces.values())).image
        arch = archive.get(image)
        rr.set_time(PLATE, sequence=p)

        # tick base: wipe the previous plate, show the new one
        rr.set_time(TICK, sequence=base)
        for m in models:
            # Not rr.Clear: newer viewers can drop a cleared tile's transform but keep
            # its image, leaving it at the origin at 2x size. Moving it away is robust.
            for path in shown_tiles[m]:
                rr.log(path, rr.Transform3D(translation=OFFSTAGE, scale=1 / SCALE))
            shown_tiles[m] = set()
            rr.log(f"gen/{m}/canvas/rating", rr.Clear(recursive=False))
            rr.log(f"gen/{m}/topk", rr.Clear(recursive=False))
        rr.log("deep/plate/verdict", rr.Clear(recursive=False))
        with Image.open(run.plate_dir / plate.name / "plate.jpg") as im:
            pw, ph = im.size
            data, ox, oy = jpeg(im, (plate_px, plate_px), centre=True)
        aspect = aspects[plate.name]
        rr.log(
            "deep/plate/image", rr.EncodedImage(contents=data, media_type="image/jpeg")
        )
        rr.log("deep/plate/info", md(_plate_md(p, len(run.plates), image, arch)))
        rr.log(
            "deep/scoreboard",
            rr.EncodedImage(
                contents=render_scoreboard(plate, models, arch), media_type="image/png"
            ),
        )
        for m in models:
            t = plate.traces[m]
            rr.log(f"deep/verdict/{m}", md(f"### {title(m)}\n*writing…*"))
            _log_seen(m, t, aspect, seen_size.get(m))
            rr.log(f"ops/latency/{m}", rr.Scalars(t.latency_s))
            rr.log(f"ops/vram/{m}", rr.Scalars(t.vram_mb / 1024))
            rr.log(f"ops/image_tokens/{m}", rr.Scalars(t.image_tokens))
            rr.log(f"ops/tokens/{m}", rr.Scalars(len(t.steps)))

        # tick base+1: the pre-filled scaffold appears, greyed out
        rr.set_time(TICK, sequence=base + 1)
        segs = {m: layout(plate.traces[m]) for m in models}
        for m in models:
            for j, seg in enumerate(s for s in segs[m] if s.step < 0):
                _tile(f"gen/{m}/canvas/t/prefill_{j}", seg, None, DIM)
                shown_tiles[m].add(f"gen/{m}/canvas/t/prefill_{j}")

        # one tick per generated token, all models in parallel
        longest = max(len(t.steps) for t in plate.traces.values())
        by_step = {m: _group(segs[m]) for m in models}
        for s in range(longest + 1):
            rr.set_time(TICK, sequence=base + 2 + s)
            for m in models:
                t = plate.traces[m]
                if s < len(t.steps):
                    step = t.steps[s]
                    bg = prob_colour(step.prob)
                    fg = text_on(bg)
                    for j, seg in enumerate(by_step[m].get(s, [])):
                        _tile(f"gen/{m}/canvas/t/{s}_{j}", seg, bg, fg)
                        shown_tiles[m].add(f"gen/{m}/canvas/t/{s}_{j}")
                    if by_step[m].get(s):
                        rr.log(
                            f"gen/{m}/canvas/cursor",
                            _box(by_step[m][s], colour(m), 2.0),
                        )
                        if s == 0:
                            rr.log(
                                f"gen/{m}/canvas/rating",
                                _box(by_step[m][s], (255, 255, 255), 1.0),
                            )
                    rr.log(
                        f"gen/{m}/topk",
                        rr.EncodedImage(
                            contents=render_topk(m, step, len(t.steps)),
                            media_type="image/png",
                        ),
                    )
                    rr.log(f"curves/confidence/{m}", rr.Scalars(step.prob))
                    rr.log(
                        f"curves/surprise/{m}",
                        rr.Scalars(-np.log2(max(step.prob, 1e-9))),
                    )
                elif s == len(t.steps):
                    rr.log(f"gen/{m}/canvas/cursor", rr.Clear(recursive=False))
                    _log_verdict(m, t, arch)

        # everyone is done: the vote
        end = base + 2 + longest + 1
        rr.set_time(TICK, sequence=end)
        votes = [normalise(plate.traces[m].rating.rating) for m in models]
        vote = calculate_majority_vote(votes)
        rr.log(
            "deep/plate/verdict",
            rr.Boxes2D(
                mins=[[ox, oy]],
                sizes=[[pw, ph]],
                colors=[RATING_COLOURS[vote]],
                labels=[f"vote: {vote}"],
                radii=[-5.0],
            ),
        )
        base = end + PADDING
    rr.reset_time()
    return starts


def _group(segs: list[Seg]) -> dict[int, list[Seg]]:
    out: dict[int, list[Seg]] = {}
    for s in segs:
        if s.step >= 0:
            out.setdefault(s.step, []).append(s)
    return out


def _plate_md(p: int, n: int, image: str, arch: Row | None) -> str:
    lines = [f"### Plate {p + 1} / {n}", f"`{image}`"]
    if arch:
        lines += [
            "",
            f"**In the archive run:** {pattern(arch)}. Vote: **{arch.majority}**.",
            "",
            "| model | archive rating |",
            "|---|---|",
        ]
        lines += [f"| {title(m)} | {r} |" for m, r in arch.ratings.items()]
    lines += ["", "Press ▶ (or space) to watch the models write."]
    return "\n".join(lines)


def _log_verdict(m: str, t: Trace, arch: Row | None) -> None:
    r = t.rating
    rating = normalise(r.rating)
    lines = [f"### {title(m)}: **{rating}**", f"*{r.category}*", "", r.rationale]
    if "}" not in t.text:
        lines += [
            "",
            f"⚠ **Cut off after {len(t.steps)} tokens.** The JSON was never closed, "
            "so the parser cannot read it and the rating becomes *Unknown*.",
        ]
    if arch is not None and arch.ratings[m] != rating:
        lines += [
            "",
            f"↔ In the archive run this model said **{arch.ratings[m]}** "
            f"({'sampled' if m == 'lavaguard' else 'same settings'}; here: greedy).",
        ]
    uncertain = sorted(t.steps, key=lambda s: s.prob)[:3]
    lines += [
        "",
        "**Least certain choices:** "
        + ", ".join(f"`{shown(s.token)}` ({s.prob:.2f})" for s in uncertain),
    ]
    lines += [
        "",
        f"`{len(t.steps)} tokens · {t.latency_s:.1f}s · {t.vram_mb / 1024:.1f} GB VRAM`",
    ]
    rr.log(f"deep/verdict/{m}", md("\n".join(lines)))


def _log_seen(
    m: str, t: Trace, aspect: float | None = None, size: tuple[int, int] | None = None
) -> None:
    if not t.seen:
        return
    pic = mosaic(t.seen, aspect)
    data, _, _ = jpeg(pic, size or pic.shape[1::-1], centre=False)
    rr.log(
        f"deep/seen/{m}/image", rr.EncodedImage(contents=data, media_type="image/jpeg")
    )
    if len(t.seen) == 1 and t.seen[0].patch_size:
        rr.log(
            f"deep/seen/{m}/patches",
            rr.LineStrips2D(
                patch_grid(pic.shape[:2], t.seen[0].patch_size),
                colors=[(255, 255, 255, 60)],
                radii=[-0.5],
            ),
        )
    else:
        rr.log(f"deep/seen/{m}/patches", rr.Clear(recursive=False))
    h, w = pic.shape[:2]
    rr.log(
        f"deep/seen/{m}/stats",
        md(
            f"### {title(m)} sees\n\n"
            + (
                f"- one overview image (left) plus {len(t.seen) - 1} high-resolution crops "
                f"(right), each {t.seen[1].pixels.shape[1]}px square\n"
                if len(t.seen) > 2
                else f"- one image, resized to {w}x{h}px\n"
            )
            + f"- **{t.image_tokens:,}** image tokens out of {t.prompt_tokens:,} prompt tokens "
            f"({t.image_tokens / max(1, t.prompt_tokens):.0%})\n"
            + (
                f"- patch size {t.seen[0].patch_size}px (grid overlaid)\n"
                if t.seen[0].patch_size and len(t.seen) == 1
                else ""
            )
        ),
    )
