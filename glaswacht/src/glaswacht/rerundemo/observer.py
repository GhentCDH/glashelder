"""Layer C: stream a live `glaswacht run` into Rerun via the orchestrator's observer hooks."""

import logging
from pathlib import Path

import rerun as rr

from glaswacht.models.safety_model import Rating
from glaswacht.orchestrator import Skip
from glaswacht.rerundemo import policy, sink
from glaswacht.rerundemo.blueprint import build_blueprint
from glaswacht.rerundemo.data import SEVERITY, Row, normalise
from glaswacht.rerundemo.replay import TIMELINE, info_md, rationale_md, thumbnail
from glaswacht.rerundemo.style import RATING_COLOURS, colour, md, title

logger = logging.getLogger(__name__)


class RerunObserver:
    """Plugs into `Orchestrator.add_observer`. Models run one after another, so each
    model's pass restarts the `image_idx` timeline; entities are per model."""

    def __init__(
        self,
        models: tuple[str, ...],
        out: Path | None = None,
        connect: str | None = None,
        thumb: int = 480,
        blur_flagged: bool = False,
    ) -> None:
        self.models = models
        self.thumb = thumb
        self.blur_flagged = blur_flagged
        self._index: dict[str, int] = {}
        self._paths: dict[str, Path] = {}
        self._unsafe = dict.fromkeys(models, 0)
        self._seen = dict.fromkeys(models, 0)
        self._ratings: dict[str, dict[str, Rating]] = {}
        blueprint = build_blueprint(models)
        sink.init(out, connect, blueprint)
        policy.log_static(models)
        for m in models:
            rr.log(
                f"stats/unsafe_rate/{m}",
                rr.SeriesLines(colors=[colour(m)], names=title(m), widths=2.0),
                static=True,
            )
            rr.log(
                f"stats/severity/{m}",
                rr.SeriesPoints(colors=[colour(m)], names=title(m), marker_sizes=3.0),
                static=True,
            )

    def on_state(self, model: str, changes: dict) -> None:
        status = changes.get("status")
        if status is not None:
            rr.log(
                "ops/lifecycle",
                rr.TextLog(f"{title(model)}: {status.value}", level="INFO"),
            )
        if "current_task" in changes and str(changes["current_task"]).startswith(
            "Skipped"
        ):
            rr.log("ops/lifecycle", rr.TextLog(changes["current_task"], level="WARN"))

    def on_rating(self, model: str, idx: int, path: str, rating: Rating) -> None:
        name = Path(path).name
        self._index[name] = idx
        self._paths[name] = Path(path)
        self._ratings.setdefault(name, {})[model] = rating
        cls = normalise(rating.rating)
        self._seen[model] += 1
        self._unsafe[model] += cls == "Unsafe"
        rr.set_time(TIMELINE, sequence=idx)
        rr.log(
            f"models/{model}/rationale", md(rationale_md(model, cls, rating.rationale))
        )
        rr.log(f"stats/severity/{model}", rr.Scalars(SEVERITY[cls]))
        rr.log(
            f"stats/unsafe_rate/{model}",
            rr.Scalars(self._unsafe[model] / self._seen[model]),
        )
        try:
            data, _, _ = thumbnail(Path(path), self.thumb, 70, False)
            rr.log(
                "plate/image", rr.EncodedImage(contents=data, media_type="image/jpeg")
            )
        except Exception as exc:
            logger.warning("no thumbnail for %s: %s", name, exc)

    def on_vote(self, image: str, vote: str, ratings: dict[str, str]) -> None:
        idx = self._index.get(image)
        if idx is None:
            return
        rr.set_time(TIMELINE, sequence=idx)
        row = Row(
            image,
            {m: normalise(r) for m, r in ratings.items()},
            {m: self._ratings[image][m].rationale for m in ratings},
            vote,
        )
        try:
            _, w, h = thumbnail(
                self._paths[image], self.thumb, 70, self.blur_flagged and row.flagged
            )
            rr.log(
                "plate/verdict",
                rr.Boxes2D(
                    mins=[[0, 0]],
                    sizes=[[w, h]],
                    colors=[RATING_COLOURS[vote]],
                    labels=[vote],
                    radii=[-6.0],
                ),
            )
        except Exception:
            logger.warning("no verdict box for %s", image, exc_info=True)
        rr.log("plate/info", md(info_md(row, tuple(row.ratings))))

    def on_skip(self, skip: Skip) -> None:
        rr.log(
            "ops/lifecycle",
            rr.TextLog(
                f"{skip.model} skipped {skip.image}: {skip.error}", level="WARN"
            ),
        )

    def close(self) -> None:
        rr.disconnect()
