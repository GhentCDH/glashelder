import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console

from metadata_extraction.extract_metadata import (
    candidate_phrases_for_record,
    search_wikidata,
)

log = logging.getLogger("metadata_extraction")

# be polite to the (unauthenticated) Wikidata API
_REQUEST_DELAY = 0.2


@click.group()
def metadata_extraction():
    pass


@metadata_extraction.command()
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of OCR json files (each with 'meta'/'content' arrays)",
)
def candidates(data: Path):
    """Print candidate entity phrases extracted from each file's OCR text."""
    console = Console()
    for path in sorted(data.glob("*.json")):
        record = json.loads(path.read_text())
        phrases = candidate_phrases_for_record(record)
        if phrases:
            console.print(f"[bold]{path.stem}[/bold]: {phrases}")


@metadata_extraction.command()
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of OCR json files (each with 'meta'/'content' arrays)",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write per-file Wikidata match jsons",
)
@click.option(
    "--lang", default="en", show_default=True, help="Wikidata search language"
)
def link(data: Path, output: Path, lang: str):
    """Extract candidate phrases and look up each one's best Wikidata match."""
    output.mkdir(exist_ok=True)
    for path in sorted(data.glob("*.json")):
        record = json.loads(path.read_text())
        phrases = candidate_phrases_for_record(record)
        if not phrases:
            continue

        log.info(f"{path.stem}: {len(phrases)} candidate(s)")
        results = {}
        for phrase in phrases:
            matches = search_wikidata(phrase, lang=lang, limit=1)
            if matches:
                results[phrase] = asdict(matches[0])
            time.sleep(_REQUEST_DELAY)

        if results:
            out = output / f"{path.stem}.json"
            out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            log.info(f"wrote {str(out)!r}")
