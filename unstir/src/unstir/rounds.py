"""A round of pairs to label, from the candidates and scores in a match store.

No matching runs: the pairs come from the store, without the pairs that already
have a label. Two kinds of pairs:

1. Plates with more than one strong reference (`f_inliers` at least `strong`).
   Their labels show if these are photos of one church or look-alikes. Whole
   plates are taken, so every strong reference of a plate gets a label.
2. Pairs in the `band` of `f_inliers` where match and no match can mix, spread
   evenly over it in steps of `step`.
"""

import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path

from common.errors import UserError
from common.fs import atomic_write

from .labels import Key, Labels

logger = logging.getLogger(__name__)


def sample(
    rows: list[dict], n: int, rng: random.Random, key: Callable[[dict], int]
) -> list[dict]:
    """`n` pairs, taken in turn from every value of `key`, highest first."""
    groups_by: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups_by[key(row)].append(row)
    groups = [rng.sample(g, len(g)) for _, g in sorted(groups_by.items(), reverse=True)]
    turns = (row for turn in zip_longest(*groups) for row in turn if row is not None)
    return list(islice(turns, n))


@dataclass(frozen=True)
class Plan:
    size: int
    multi: int
    """At most this many pairs of plates with more than one strong reference."""
    strong: int
    band: tuple[int, int]
    step: int
    seed: int


def labelled(paths: list[Path]) -> set[Key]:
    return {key for path in paths for key in Labels(path).labels}


def multi_plates(
    rows: list[dict], strong: int, limit: int, rng: random.Random
) -> list[dict]:
    """Whole plates with more than one strong reference, until `limit` pairs."""
    by_plate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["f_inliers"] >= strong:
            by_plate[row["plate"]].append(row)
    plates = [p for p in sorted(by_plate) if len(by_plate[p]) > 1]
    rng.shuffle(plates)
    chosen: list[dict] = []
    for plate in plates:
        if len(chosen) + len(by_plate[plate]) > limit:
            continue
        chosen += by_plate[plate]
    return chosen


def run(rows: list[dict], labels: list[Path], output: Path, plan: Plan) -> None:
    rng = random.Random(plan.seed)
    done = labelled(labels)
    rows = [row for row in rows if (row["plate"], row["reference"]) not in done]
    logger.info(
        f"{len(done):,} labelled pairs in {len(labels)} files, "
        f"{len(rows):,} candidate pairs without a label"
    )

    multi = multi_plates(rows, plan.strong, min(plan.multi, plan.size), rng)
    taken = {(r["plate"], r["reference"]) for r in multi}
    low, high = plan.band
    band = [
        r
        for r in rows
        if low <= r["f_inliers"] < high and (r["plate"], r["reference"]) not in taken
    ]
    spread = sample(
        band, plan.size - len(multi), rng, key=lambda r: r["f_inliers"] // plan.step
    )
    chosen = multi + spread
    if not chosen:
        raise UserError("no pairs without a label match these settings")

    output.mkdir(parents=True, exist_ok=True)
    target = output / "pairs.jsonl"
    if target.exists():
        raise UserError(f"{target} exists already, choose another --output")
    atomic_write(target, "".join(json.dumps(r) + "\n" for r in chosen))
    meta = {
        "labels": [str(p) for p in labels],
        "size": plan.size,
        "multi": plan.multi,
        "strong": plan.strong,
        "band": list(plan.band),
        "step": plan.step,
        "seed": plan.seed,
        "multi_pairs": len(multi),
        "multi_plates": len({r["plate"] for r in multi}),
        "band_pairs": len(spread),
    }
    atomic_write(output / "meta.json", json.dumps(meta, indent=2) + "\n")
    logger.info(
        f"{len(chosen)} pairs in {target}: {len(multi)} of "
        f"{meta['multi_plates']} plates with more than one strong reference, "
        f"{len(spread)} from f_inliers {low} to {high}"
    )
