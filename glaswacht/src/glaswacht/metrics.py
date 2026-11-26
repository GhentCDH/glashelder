"""How long a run spends loading models and assessing images, and how much
GPU memory it takes: to size --max-images to a job's time limit."""

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

_GIB = 2**30


def gpu_gib() -> float | None:
    """GPU memory torch holds now, in GiB; None without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / _GIB


def _peak_gpu_gib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / _GIB


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class ModelMetrics:
    load_seconds: float | None = None
    # GPU memory the loaded model takes
    gpu_gib: float | None = None
    # one per image it assessed; failed attempts are counted, not timed
    assess_seconds: list[float] = field(default_factory=list)
    failures: int = 0

    def stats(self) -> dict:
        s = self.assess_seconds
        return {
            "load_seconds": _round(self.load_seconds),
            "gpu_gib": _round(self.gpu_gib),
            "assessed": len(s),
            "failed": self.failures,
            "seconds_total": _round(sum(s)),
            "seconds_mean": _round(statistics.fmean(s)) if s else None,
            "seconds_median": _round(statistics.median(s)) if s else None,
            "seconds_min": _round(min(s)) if s else None,
            "seconds_max": _round(max(s)) if s else None,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


class RunMetrics:
    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.models: dict[str, ModelMetrics] = {}
        # time spent going through images, without loading and unloading
        self.processing_seconds = 0.0
        self.wall_seconds = 0.0
        self.images_finished = 0
        self.peak_gpu_gib: float | None = None
        _reset_peak()

    def model(self, name: str) -> ModelMetrics:
        return self.models.setdefault(name, ModelMetrics())

    def finish(self, images_finished: int) -> None:
        self.wall_seconds = time.perf_counter() - self._started
        self.images_finished = images_finished
        self.peak_gpu_gib = _peak_gpu_gib()

    def seconds_per_image(self) -> float | None:
        """Assessing one image with every model, on average."""
        means = [
            statistics.fmean(m.assess_seconds)
            for m in self.models.values()
            if m.assess_seconds
        ]
        return sum(means) if means else None

    def as_dict(self) -> dict:
        per_image = self.seconds_per_image()
        return {
            "wall_seconds": _round(self.wall_seconds),
            "load_seconds": _round(
                sum(m.load_seconds or 0 for m in self.models.values())
            ),
            "processing_seconds": _round(self.processing_seconds),
            "images_finished": self.images_finished,
            "seconds_per_image": _round(per_image),
            "images_per_hour": round(3600 / per_image) if per_image else None,
            "peak_gpu_gib": _round(self.peak_gpu_gib),
            "models": {name: m.stats() for name, m in self.models.items()},
        }

    def report(self) -> list[str]:
        """The metrics as lines for the log and the job output."""
        d = self.as_dict()
        lines = [
            f"Time: {_duration(d['wall_seconds'])} in all, "
            f"{_duration(d['load_seconds'])} loading models, "
            f"{_duration(d['processing_seconds'])} assessing images",
        ]
        for name, m in d["models"].items():
            parts = [f"  {name:<11}"]
            if m["load_seconds"] is not None:
                memory = f", {m['gpu_gib']:.1f} GiB" if m["gpu_gib"] is not None else ""
                parts.append(f"load {m['load_seconds']:.1f} s{memory}")
            if m["assessed"]:
                parts.append(
                    f"{m['assessed']} images, {m['seconds_mean']:.1f} s each "
                    f"(median {m['seconds_median']:.1f}, "
                    f"min {m['seconds_min']:.1f}, max {m['seconds_max']:.1f})"
                )
            if m["failed"]:
                parts.append(f"{m['failed']} failed")
            lines.append(" | ".join(parts))
        if d["seconds_per_image"] is not None:
            lines.append(
                f"Per image, all models: {d['seconds_per_image']:.1f} s, "
                f"about {d['images_per_hour']} images per hour after loading"
            )
        if d["peak_gpu_gib"] is not None:
            lines.append(f"Peak GPU memory: {d['peak_gpu_gib']:.1f} GiB")
        return lines

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def _duration(seconds: float | None) -> str:
    seconds = round(seconds or 0)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
