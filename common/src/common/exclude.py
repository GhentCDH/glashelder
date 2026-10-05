"""Leaving parts of the collection out, based on CSV exports of it.

The collection metadata exports (`etnografie.csv`, `medisch.csv`, ...) have a `pid`
column, and images are named after it: `m61bk56v14_tiff.jpg` belongs to pid
`m61bk56v14`. Giving such a CSV leaves every image of its rows out.

Any CSV with a `pid` column will do, so a file that excludes one image is:

    pid
    m61bk56v14

and is used as `cluster cluster -x one-image.csv`. Other columns are ignored, and
`--exclude-column` names another column than `pid`.
"""

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PID_COLUMN = "pid"


def read_pids(paths: Iterable[str | Path], column: str = PID_COLUMN) -> set[str]:
    """The values of `column` over all of the CSV files, blanks left out."""
    pids: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(f"{path} has no `{column}` column")
            found = {row[column].strip() for row in reader if row[column]}
        logger.info(f"{path}: {len(found):,} {column} values")
        pids |= found
    return pids


def pid_of(ref: str) -> str:
    """The pid an image ref belongs to: the file name up to its first underscore."""
    return Path(ref).stem.split("_", 1)[0]


def excluded_mask(
    refs: Sequence[str], paths: Iterable[str | Path], column: str = PID_COLUMN
) -> np.ndarray:
    """`(len(refs),)` bool array, True where the ref belongs to a row of the CSV files."""
    pids = read_pids(paths, column)
    return np.fromiter((pid_of(ref) in pids for ref in refs), bool, count=len(refs))
