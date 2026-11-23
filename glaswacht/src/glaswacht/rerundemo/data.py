"""Results loading and statistics. Pure numpy, no Rerun, so it is unit-testable."""

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glaswacht.models.utils import calculate_majority_vote

# canonical rating classes, ordered by severity
CLASSES = ("Safe", "Potentially Problematic", "Unsafe", "Unknown")
SEVERITY = {c: i for i, c in enumerate(CLASSES)}
DEFAULT_MODELS = ("lavaguard", "qwen3", "pixtral")


def normalise(rating: str) -> str:
    """Map anything the parser produced onto the four display classes."""
    return rating if rating in SEVERITY else "Unknown"


@dataclass
class Row:
    image: str
    ratings: dict[str, str]
    rationales: dict[str, str]
    majority: str = ""

    @property
    def flagged(self) -> bool:
        return self.majority in ("Unsafe", "Potentially Problematic")


@dataclass
class Stats:
    models: tuple[str, ...]
    n: int
    # per model: severity class index per image, shape (n,)
    severity: dict[str, np.ndarray]
    majority: np.ndarray
    unsafe_rate: dict[str, np.ndarray]
    unknown_count: dict[str, np.ndarray]
    agree: dict[tuple[str, str], np.ndarray]
    flagged_total: np.ndarray
    all_agree: np.ndarray
    confusion: dict[tuple[str, str], np.ndarray]
    distribution: dict[str, np.ndarray]
    review_order: list[int] = field(
        default_factory=list
    )  # image idx, most contested first


def load_results(
    csv_path: str | Path, models: tuple[str, ...] = DEFAULT_MODELS
) -> list[Row]:
    """Read a results CSV in either the `<model>_rating` or `model_N_rating` schema."""
    csv.field_size_limit(2**31 - 1)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        if all(f"{m}_rating" in header for m in models):
            keys = {m: (f"{m}_rating", f"{m}_rationale") for m in models}
        elif all(f"model_{i + 1}_rating" in header for i in range(len(models))):
            keys = {
                m: (f"model_{i + 1}_rating", f"model_{i + 1}_rationale")
                for i, m in enumerate(models)
            }
        else:
            raise ValueError(f"unrecognised results columns in {csv_path}: {header}")
        rows = []
        for rec in reader:
            ratings = {m: normalise(rec[k[0]]) for m, k in keys.items()}
            rows.append(
                Row(
                    image=rec["image"],
                    ratings=ratings,
                    rationales={m: rec[k[1]] for m, k in keys.items()},
                    majority=calculate_majority_vote(list(ratings.values())),
                )
            )
    return rows


def contest_score(row: Row) -> float:
    """How worth a human look a row is: spread of verdicts, then how severe."""
    sev = [SEVERITY[r] for r in row.ratings.values()]
    known = [s for s in sev if s != SEVERITY["Unknown"]]
    spread = (max(known) - min(known)) if known else 0
    unknown = sum(s == SEVERITY["Unknown"] for s in sev)
    distinct = len(set(sev)) - 1  # a three-way split beats a lone dissenter
    return (
        spread * 10
        + distinct * 2
        + unknown * 5
        + sum(s == SEVERITY["Unsafe"] for s in sev)
    )


def compute_stats(rows: list[Row], models: tuple[str, ...] = DEFAULT_MODELS) -> Stats:
    n = len(rows)
    steps = np.arange(1, n + 1)
    severity = {
        m: np.array([SEVERITY[r.ratings[m]] for r in rows], dtype=np.int8)
        for m in models
    }
    majority = np.array([SEVERITY[r.majority] for r in rows], dtype=np.int8)
    unsafe = SEVERITY["Unsafe"]
    unknown = SEVERITY["Unknown"]

    pairs = list(itertools.combinations(models, 2))
    confusion = {}
    for a, b in pairs:
        mat = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        np.add.at(mat, (severity[a], severity[b]), 1)
        confusion[(a, b)] = mat

    contested = sorted(
        (i for i, r in enumerate(rows) if contest_score(r) > 0),
        key=lambda i: (-contest_score(rows[i]), i),
    )
    return Stats(
        models=models,
        n=n,
        severity=severity,
        majority=majority,
        unsafe_rate={m: np.cumsum(severity[m] == unsafe) / steps for m in models},
        unknown_count={m: np.cumsum(severity[m] == unknown) for m in models},
        agree={(a, b): np.cumsum(severity[a] == severity[b]) / steps for a, b in pairs},
        flagged_total=np.cumsum(majority >= SEVERITY["Potentially Problematic"])
        * (majority.size > 0),
        all_agree=np.cumsum(
            np.all([severity[m] == severity[models[0]] for m in models], axis=0)
        )
        / steps,
        confusion=confusion,
        distribution={
            m: np.bincount(severity[m], minlength=len(CLASSES)) for m in models
        },
        review_order=contested,
    )


def pattern(r: Row) -> str:
    """Name the verdict pattern of a plate, for picking and for listing."""
    vals = list(r.ratings.values())
    if "Unknown" in vals:
        return "parse failure"
    if len(set(vals)) == 1:
        return f"unanimous {vals[0]}"
    if len(set(vals)) == len(vals):
        return "split three ways"
    lone = next(m for m, v in r.ratings.items() if vals.count(v) == 1)
    return f"{lone} alone says {r.ratings[lone]}"


def pick_diverse(rows: list[Row], count: int = 25, seed: int = 7) -> list[Row]:
    """A demo set covering every verdict pattern: rarest patterns first, round-robin."""
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[Row]] = {}
    for r in rows:
        buckets.setdefault(pattern(r), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets, key=lambda k: len(buckets[k]))
    picked: list[Row] = []
    while len(picked) < count and any(buckets.values()):
        for k in order:
            if buckets[k] and len(picked) < count:
                picked.append(buckets[k].pop())
    return picked
