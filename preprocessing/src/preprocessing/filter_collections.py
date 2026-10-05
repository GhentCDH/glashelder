#!/usr/bin/env python3
"""List or exclude collections ("Het geheel waartoe de glasplaat behoort") in a CSV.

Examples:
    python filter_collections.py --list
    python filter_collections.py --include "Collectie A" --name only_a.csv
    python filter_collections.py --exclude "Collectie A" --exclude "Collectie B" --name filtered.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

COLUMN = "Het geheel waartoe de glasplaat behoort"
EMPTY_LABEL = "(empty)"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        "-i",
        default="",
        help="input CSV (default: %(default)s)",
    )
    p.add_argument(
        "--list", action="store_true", help="list all collections with row counts"
    )
    p.add_argument(
        "--exclude",
        "-e",
        action="append",
        default=[],
        metavar="COLLECTION",
        help=f"collection to remove (repeatable; use '{EMPTY_LABEL}' for rows without a collection)",
    )
    p.add_argument(
        "--include",
        "-I",
        action="append",
        default=[],
        metavar="COLLECTION",
        help="collection to keep exclusively (repeatable; all others are dropped)",
    )
    p.add_argument(
        "--name", "-n", help="output CSV name (default: <input>_filtered.csv)"
    )
    args = p.parse_args()

    if not args.list and not args.exclude and not args.include:
        p.error("use --list, --include or --exclude")

    src = Path(args.input)
    with src.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if COLUMN not in fieldnames:
            sys.exit(f"Column '{COLUMN}' not found in {src}")
        rows = list(reader)

    def label(row: dict) -> str:
        return (row[COLUMN] or "").strip() or EMPTY_LABEL

    counts = Counter(label(r) for r in rows)

    if args.list:
        for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"{n:>8}  {name}")
        print(f"\n{len(counts)} collections, {len(rows)} rows")
        if not args.exclude and not args.include:
            return 0

    unknown = [e for e in args.exclude + args.include if e not in counts]
    if unknown:
        sys.exit(
            "Unknown collection(s): "
            + ", ".join(repr(u) for u in unknown)
            + " (see --list)"
        )

    excluded = set(args.exclude)
    included = set(args.include)
    kept = [
        r
        for r in rows
        if label(r) not in excluded and (not included or label(r) in included)
    ]

    out = Path(args.name) if args.name else src.with_name(f"{src.stem}_filtered.csv")
    if out.suffix == "":
        out = out.with_suffix(".csv")
    if out.resolve() == src.resolve():
        sys.exit("Output would overwrite the input; choose a different --name")

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Removed {len(rows) - len(kept)} rows, wrote {len(kept)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
