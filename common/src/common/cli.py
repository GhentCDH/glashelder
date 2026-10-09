"""Click options that several commands share."""

from collections.abc import Callable
from functools import reduce
from typing import Any

import click

Decorator = Callable[[Any], Any]


def options(*decorators: Decorator) -> Decorator:
    """All `decorators` as one, in the order given: the first comes first in --help."""

    def apply(command: Any) -> Any:
        return reduce(
            lambda wrapped, option: option(wrapped), reversed(decorators), command
        )

    return apply


def serve_options(port: int) -> Decorator:
    """The options of a serve command, which `run_app` takes as keyword arguments:
    `host`, `port`, `auth`, `debug` and `open_browser`."""
    return options(
        click.option(
            "--host", default="0.0.0.0", show_default=True, help="Host to bind to"
        ),
        click.option("--port", default=port, show_default=True, help="Port to bind to"),
        click.option(
            "--auth/--no-auth",
            default=True,
            show_default=True,
            help="Require the `secret` config value: open /?secret=<secret> once "
            "to get a session cookie",
        ),
        click.option(
            "--debug",
            is_flag=True,
            default=False,
            show_default=True,
            help="Run Flask with its debugger and auto-reloader",
        ),
        click.option(
            "--open/--no-open",
            "open_browser",
            default=True,
            show_default=True,
            help="Open the site in a browser on startup",
        ),
    )
