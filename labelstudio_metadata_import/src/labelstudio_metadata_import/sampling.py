"""Random sampling of images from a folder, for spot-checking a batch in
Label Studio without running the full Glaswacht assessment.
"""

from __future__ import annotations

import csv
import secrets
from dataclasses import dataclass
from pathlib import Path

from labelstudio_metadata_import.importer import EXTENSIONS

# Defaults for the GLAS metadata export: column U is the collection name,
# 'pid' identifies the image (filenames on disk are typically "{pid}_....ext").
DEFAULT_GROUP_COLUMN = "Het geheel waartoe de glasplaat behoort"
DEFAULT_ID_COLUMN = "pid"


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def resolve_sample_size(total: int, count: int | None, percent: float | None) -> int:
    """Resolve --count/--percent to an absolute sample size, clamped to [0, total]."""
    n = count if percent is None else round(total * percent / 100)
    return max(0, min(n, total))


def random_sample(images: list[Path], size: int) -> list[Path]:
    """Select SIZE images uniformly at random.

    Uses secrets.SystemRandom, which draws from the OS's CSPRNG (os.urandom)
    rather than the default seeded Mersenne Twister — the result is not
    reproducible and cannot be seeded, by design.
    """
    return secrets.SystemRandom().sample(images, size)


@dataclass
class GroupSampleReport:
    group: str
    rows: int
    matched: int
    selected: int


def read_id_groups(
    csv_path: str | Path,
    group_column: str = DEFAULT_GROUP_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
) -> dict[str, list[str]]:
    """Group row identifiers (ID_COLUMN) by GROUP_COLUMN value."""
    groups: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            identifier = (row.get(id_column) or "").strip()
            if not identifier:
                continue
            group = (row.get(group_column) or "").strip() or "(none)"
            groups.setdefault(group, []).append(identifier)
    return groups


def index_images_by_id(images: list[Path]) -> dict[str, Path]:
    """Index images by identifier, matching either the whole filename stem
    ('abc123') or the part before the first underscore ('abc123' out of
    'abc123_tiff.jpg') — covers the common {id}_suffix.ext convention without
    assuming one fixed suffix or extension.
    """
    index: dict[str, Path] = {}
    for p in images:
        stem = p.stem
        index.setdefault(stem, p)
        index.setdefault(stem.split("_", 1)[0], p)
    return index


def resolve_group_sizes_from_total(
    matched_counts: dict[str, int], total: int
) -> dict[str, int]:
    """Distribute TOTAL as evenly as possible across groups, capped by each
    group's own availability (MATCHED_COUNTS). Groups that run out get their
    shortfall redistributed to the remaining groups (water-filling), so the
    full TOTAL is used whenever enough images exist somewhere to cover it.

    Which groups receive the (possibly uneven) remainder is decided by
    secrets.SystemRandom, not iteration order, so it isn't biased towards
    e.g. alphabetically-first collections.
    """
    rng = secrets.SystemRandom()
    remaining = {g: n for g, n in matched_counts.items() if n > 0}
    sizes = dict.fromkeys(matched_counts, 0)
    budget = max(0, min(total, sum(matched_counts.values())))

    while budget > 0 and remaining:
        order = list(remaining)
        rng.shuffle(order)
        share = max(1, budget // len(remaining))
        for group in order:
            if budget <= 0:
                break
            take = min(share, remaining[group], budget)
            if take <= 0:
                continue
            sizes[group] += take
            remaining[group] -= take
            budget -= take
            if remaining[group] == 0:
                del remaining[group]

    return sizes


def sample_by_group(
    images: list[Path],
    groups: dict[str, list[str]],
    count: int | None,
    percent: float | None,
) -> tuple[list[Path], list[GroupSampleReport]]:
    """Randomly sample a total of --count/--percent images, spread as evenly
    as possible across every group (collection), so small and large
    collections end up roughly equally represented instead of the sample
    being dominated by the biggest one.

    --percent is a percentage of the total matched pool (across all groups),
    not per group. Only identifiers that actually match a file in IMAGES are
    eligible; rows whose image isn't present in the given folder are
    reported but skipped.
    """
    index = index_images_by_id(images)
    matched_by_group = {
        group: [index[i] for i in identifiers if i in index]
        for group, identifiers in groups.items()
    }

    total_matched = sum(len(matched) for matched in matched_by_group.values())
    total = resolve_sample_size(total_matched, count, percent)
    sizes = resolve_group_sizes_from_total(
        {g: len(matched) for g, matched in matched_by_group.items()}, total
    )

    chosen: list[Path] = []
    reports: list[GroupSampleReport] = []
    for group, identifiers in sorted(groups.items()):
        matched = matched_by_group[group]
        selected = random_sample(matched, sizes[group])
        chosen.extend(selected)
        reports.append(
            GroupSampleReport(group, len(identifiers), len(matched), len(selected))
        )

    return chosen, reports
