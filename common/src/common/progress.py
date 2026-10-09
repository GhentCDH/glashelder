"""Progress bars that look the same in every command."""

import math
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def columns() -> tuple[ProgressColumn, ...]:
    """The usual layout: spinner, description, bar, count, percentage and times."""
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def progress(
    *custom: str | ProgressColumn,
    stderr: bool = False,
    transient: bool = False,
    **kwargs: Any,
) -> Progress:
    """A `Progress` with the usual columns, or with `custom` ones when given.

    `stderr` draws on standard error, so the bars stay out of piped output. Pass
    `console` instead to share one with a logging handler. Other keyword arguments
    go to `Progress`.
    """
    if stderr:
        kwargs.setdefault("console", Console(stderr=True))
    return Progress(*(custom or columns()), transient=transient, **kwargs)


def format_bytes(size: float) -> str:
    power = min(3, int(math.log10(size) // 3)) if size >= 1 else 0
    unit = ("B", "KB", "MB", "GB")[power]
    return f"{size / 1000**power:.1f} {unit}" if power else f"{size:.0f} B"
