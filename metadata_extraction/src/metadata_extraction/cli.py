import json
import time
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console

from metadata_extraction.extract_metadata import (
    candidate_phrases_for_record,
    search_wikidata,
)

# be polite to the (unauthenticated) Wikidata API
_REQUEST_DELAY = 0.2


@click.group()
def metadata_extraction():
    pass


def _iter_json_files(data: Path, recursive: bool):
    pattern = "**/*.json" if recursive else "*.json"
    for path in sorted(data.glob(pattern)):
        if path.stem != "template":
            yield path


@metadata_extraction.command()
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of typed OCR json files",
)
@click.option(
    "-r", "--recursive", is_flag=True, default=False, help="Also scan subdirectories"
)
def candidates(data: Path, recursive: bool):
    """Print candidate entity phrases extracted from each file's OCR text."""
    console = Console()
    for path in _iter_json_files(data, recursive):
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
    help="Directory of typed OCR json files",
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
@click.option(
    "-r", "--recursive", is_flag=True, default=False, help="Also scan subdirectories"
)
def link(data: Path, output: Path, lang: str, recursive: bool):
    """Extract candidate phrases and look up each one's best Wikidata match."""
    console = Console()
    output.mkdir(exist_ok=True)
    for path in _iter_json_files(data, recursive):
        record = json.loads(path.read_text())
        phrases = candidate_phrases_for_record(record)
        if not phrases:
            continue

        console.print(f"[bold]{path.stem}[/bold]: {len(phrases)} candidate(s)")
        results = {}
        for phrase in phrases:
            matches = search_wikidata(phrase, lang=lang, limit=1)
            if matches:
                m = matches[0]
                console.print(f"  [green]✓[/green] {phrase!r} → {m.label!r} ({m.id}, sim={m.similarity:.2f})")
                results[phrase] = asdict(m)
            else:
                console.print(f"  [yellow]–[/yellow] {phrase!r}: no Wikidata match")
            time.sleep(_REQUEST_DELAY)

        if results:
            out = output / f"{path.stem}.json"
            out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            console.print(f"  [blue]→ wrote {out.name}[/blue]")
        else:
            console.print(f"  [red]✗ no matches found, nothing saved[/red]")
