"""One decision per plate, from the scores of an `unstir pairs` run.

A plate shows at most one building, so its candidates are compared with each
other: the best reference (most `f_inliers`) is a match when

- it has at least `min_f` fundamental matrix inliers,
- at least `min_ratio` of its matches are inliers, and
- it has at least `margin` more inliers than the next best reference.

The margin rejects look-alike churches: when two references score about the
same, neither is trusted. On the labels of rounds 1 and 2 this gave precision
0.92 and recall 0.95 with the defaults (chosen on those same labels).

Writes `decisions.csv` (every plate) and `matches.jsonl` (the matches, in the
pairs format, so `unstir serve` can show them for a check).
"""

import csv
import io
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from common.fs import atomic_write

logger = logging.getLogger(__name__)

PROVENANCE = "provenance.jsonl"


@dataclass(frozen=True)
class Rule:
    min_f: int = 150
    min_ratio: float = 0.8
    margin: int = 50


def ratio(row: dict) -> float:
    return row["f_inliers"] / max(row["matches"], 1)


def provenance(references: Path | None) -> dict[str, dict]:
    """What the scraper recorded per reference photo, by file name, if it is there."""
    path = references / PROVENANCE if references is not None else None
    if path is None or not path.is_file():
        return {}
    found = {}
    with path.open(encoding="utf-8") as file:
        for line in filter(str.strip, file):
            row = json.loads(line)
            thing = row.get("object", {})
            found[f"{row['image_id']}.jpg"] = {
                "name": thing.get("naam", ""),
                "location": thing.get("locatie_samenvatting", ""),
                "uri": thing.get("uri", ""),
            }
    return found


def decide(rows: list[dict], rule: Rule) -> list[dict]:
    by_plate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["outside"]:
            by_plate[row["plate"]].append(row)

    def one(candidates: list[dict]) -> dict:
        best, *rest = sorted(candidates, key=lambda r: -r["f_inliers"])
        second = rest[0] if rest else None
        margin = best["f_inliers"] - (second["f_inliers"] if second else 0)
        match = (
            best["f_inliers"] >= rule.min_f
            and ratio(best) >= rule.min_ratio
            and margin >= rule.margin
        )
        return {
            "plate": best["plate"],
            "decision": "match" if match else "no_match",
            "reference": best["reference"],
            "f_inliers": best["f_inliers"],
            "ratio": round(ratio(best), 3),
            "h_inliers": best["h_inliers"],
            "margin": margin,
            "rank": best["rank"],
            "second": second["reference"] if second else "",
            "second_f_inliers": second["f_inliers"] if second else 0,
        }

    return sorted(
        (one(c) for c in by_plate.values()),
        key=lambda d: (d["decision"] != "match", -d["f_inliers"]),
    )


def run(scores: Path, output: Path, rule: Rule, references: Path | None) -> None:
    decisions = decide(pq.read_table(scores).to_pylist(), rule)
    known = provenance(references)
    for d in decisions:
        d.update(known.get(Path(d["reference"]).name, {}))

    output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for d in decisions for key in d))
    text = io.StringIO()
    writer = csv.DictWriter(text, columns)
    writer.writeheader()
    writer.writerows(decisions)
    atomic_write(output / "decisions.csv", text.getvalue())

    matches = [d for d in decisions if d["decision"] == "match"]
    atomic_write(
        output / "matches.jsonl", "".join(json.dumps(d) + "\n" for d in matches)
    )
    meta = {"scores": str(scores), "rule": rule.__dict__, "plates": len(decisions)}
    atomic_write(output / "meta.json", json.dumps(meta, indent=2) + "\n")
    logger.info(
        f"{len(matches):,} of {len(decisions):,} plates match a reference. "
        f"Written to {output}/decisions.csv and matches.jsonl"
    )
