"""CLI for scraping an index of externally described artworks."""

import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .backends import registry
from .client import Stats
from .harness import combine_files, scrape
from .model import Artwork

logger = logging.getLogger(__name__)


@click.group()
def artscraper():
    """Scrape an index of artworks from external data sources."""


def progress_reporter(progress: Progress):
    """One spinner per backend, showing items, requests and rate."""
    tasks: dict[str, TaskID] = {}

    def report(name: str, _artwork: Artwork | None, stats: Stats) -> None:
        if name not in tasks:
            tasks[name] = progress.add_task(name, total=None)
        progress.update(
            tasks[name],
            completed=stats.items,
            description=f"{name}: {stats.items} artworks, "
            f"{stats.requests} requests ({stats.rate:.1f}/s), "
            f"{stats.retries} retries, {stats.errors} errors",
        )

    return report


TUI_COLUMNS = (
    "backend",
    "artworks",
    "requests",
    "req/s now",
    "req/s avg",
    "latency",
    "latency avg",
    "busy",
    "idle",
    "retries",
    "errors",
    "elapsed",
)


class TuiView:
    """A table of every backend's counters, with progress bars below it.

    `Live` calls `__rich__` on every refresh, so the numbers keep moving while
    requests are in flight, not only when an artwork arrives.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.seen: dict[str, Stats] = {}
        self.tasks: dict[str, TaskID] = {}
        self.bars = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def report(self, name: str, _artwork: Artwork | None, stats: Stats) -> None:
        self.seen[name] = stats
        if name not in self.tasks:
            # without a limit there is no known total, so the bar just pulses
            self.tasks[name] = self.bars.add_task(name, total=self.limit or None)
        self.bars.update(self.tasks[name], completed=stats.items)

    def table(self) -> Table:
        table = Table(expand=True)
        for column in TUI_COLUMNS:
            table.add_column(column, justify="left" if column == "backend" else "right")
        for name, stats in self.seen.items():
            table.add_row(
                name,
                str(stats.items),
                str(stats.requests),
                f"{stats.current_rate:.1f}",
                f"{stats.rate:.1f}",
                f"{stats.last_latency * 1000:.0f}ms",
                f"{stats.avg_latency * 1000:.0f}ms",
                str(stats.in_flight),
                str(stats.idle),
                str(stats.retries),
                str(stats.errors),
                f"{stats.elapsed:.0f}s",
            )
        return table

    def __rich__(self) -> Group:
        return Group(Panel(self.table(), title="artscraper"), self.bars)


@artscraper.command("scrape")
@click.option(
    "--backend",
    "backends",
    multiple=True,
    type=click.Choice(sorted(registry())),
    help="Backend to scrape, repeatable. Defaults to all of them.",
)
@click.option("--concurrency", default=10, help="Requests in flight per backend")
@click.option("--rate", default=5.0, help="Requests per second per backend")
@click.option(
    "--limit", default=0, help="Artworks per backend to fetch, 0 for no limit"
)
@click.option(
    "--with-images/--no-images",
    default=True,
    help="Fetch each artwork's page to find its image URL. Slow but that is the point.",
)
@click.option(
    "--jsonl",
    type=click.Path(path_type=Path),
    default=Path("artworks.jsonl"),
    help="Streamed results, also used to resume an interrupted run",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, allow_dash=True),
    default="-",
    show_default=True,
    help="Where the JSON list goes (.json, or compressed: .json.zst), '-' for stdout. "
    "Defaults to `paths.artworks` from glashelder.toml.",
)
@click.option("--tui", is_flag=True, help="Show a live table instead of progress bars")
def scrape_command(backends, concurrency, rate, limit, with_images, jsonl, output, tui):
    """Scrape the configured backends into a JSON list of artworks."""
    output = None if str(output) == "-" else output
    if output is not None and output.resolve() == jsonl.resolve():
        raise click.BadParameter(
            "the JSON list would overwrite the JSONL it is made from",
            param_hint="--output",
        )
    names = list(backends) or sorted(registry())
    console = Console(stderr=True)

    if tui:
        view = TuiView(limit)
        with Live(view, console=console, screen=True, refresh_per_second=4):
            run(names, view.report, jsonl, concurrency, rate, limit, with_images)
    else:
        columns = [
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=10),
            TimeElapsedColumn(),
        ]
        with Progress(*columns, console=console) as progress:
            run(
                names,
                progress_reporter(progress),
                jsonl,
                concurrency,
                rate,
                limit,
                with_images,
            )

    return combine_files(jsonl, path=output)


@artscraper.command("combine_jsons")
@click.argument(
    "json_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
    help="output json file path",
)
def combine_jsons_command(json_files: tuple[Path, ...], output: Path):
    """Combine one or more jsonl or json files of `Artwork`s into one json"""
    output = Path(output)
    json_files = tuple(Path(json_file) for json_file in json_files)

    if output.exists() and not click.confirm(
        f"Warning: '{output}' already exists. Do you want to overwrite it?"
    ):
        click.echo("Aborted!")
        return

    combine_files(*json_files, path=output)


def run(names, report, jsonl, concurrency, rate, limit, with_images):
    """Scrape, keeping whatever reached the JSONL if the user interrupts."""
    try:
        asyncio.run(
            scrape(
                names,
                jsonl=jsonl,
                concurrency=concurrency,
                rate=rate,
                limit=limit,
                with_images=with_images,
                report=report,
            )
        )
    except KeyboardInterrupt:
        logger.warning("interrupted, keeping what was scraped so far")
