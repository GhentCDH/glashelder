"""`glashelder config example`: write `glashelder.example.toml` from the schema in
`common.config`, or check that it still matches.

Every setting is written with its default, and the field docstrings become its
comments. A setting that is unset by default is commented out, with its type as
a placeholder. The check compares values only, so comments, order and formatting
in the file are free to change.
"""

import ast
import inspect
import json
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import click

from . import PROJECT_ROOT
from . import config as schema
from .config import Config, build

EXAMPLE_NAME = "glashelder.example.toml"

HEADER = """\
# Every setting glashelder reads, with its default. A commented-out setting is
# unset by default. Generated from `common/src/common/config.py` by
# `uv run glashelder config example`, so change the schema there, not this file.
#
# `glashelder.toml` is not committed: run `uv run glashelder setup` to write your
# own from a few plain-language questions, or copy the settings you change from
# here. Every value can be overridden from the environment, e.g.
#   GLASHELDER_PATHS__IMAGES__GLASHELDER=/mnt/other/images"""


@cache
def docstrings() -> dict[str, dict[str, str]]:
    """Per schema class, its docstring (under `""`) and those of its fields.

    Field docstrings are not kept at runtime, so they come from the source.
    """
    classes = (
        node
        for node in ast.parse(inspect.getsource(schema)).body
        if isinstance(node, ast.ClassDef)
    )
    return {
        node.name: {
            "": ast.get_docstring(node) or "",
            **{
                field.target.id: inspect.cleandoc(doc.value.value)
                for field, doc in zip(node.body, node.body[1:], strict=False)
                if isinstance(field, ast.AnnAssign)
                and isinstance(field.target, ast.Name)
                and isinstance(doc, ast.Expr)
                and isinstance(doc.value, ast.Constant)
                and isinstance(doc.value.value, str)
            },
        }
        for node in classes
    }


def comment(text: str) -> list[str]:
    return [f"# {line}".rstrip() for line in text.splitlines()]


def commented(lines: list[str]) -> list[str]:
    return [line if not line or line.startswith("#") else f"# {line}" for line in lines]


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path | str):
        return json.dumps(str(value))
    return repr(value)


def placeholder(annotation: Any) -> str:
    """`<type>` for a setting without a default, from its annotation."""
    if get_origin(annotation) in (Union, UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        annotation = options[0] if len(options) == 1 else annotation
    return (
        "<path>"
        if annotation is Path
        else f"<{getattr(annotation, '__name__', annotation)}>"
    )


def setting(name: str, value: Any, annotation: Any) -> str:
    if value is None:
        return f"# {name} = {placeholder(annotation)}"
    return f"{name} = {toml_value(value)}"


def tables(section: object, name: str = "") -> list[list[str]]:
    """The lines of `[name]`, then those of each table below it."""
    docs = docstrings().get(type(section).__name__, {})
    keys: list[list[str]] = []
    below: list[list[str]] = []
    for field in fields(section):  # ty: ignore[invalid-argument-type]
        value, path = getattr(section, field.name), f"{name}.{field.name}".lstrip(".")
        doc = comment(docs.get(field.name, ""))
        if is_dataclass(value):
            below += tables(value, path)
        elif get_origin(field.type) is dict:
            # a table of tables keyed by name: show one, commented out
            item = get_args(field.type)[1]()
            lines = [
                line for table in tables(item, f'{path}."<name>"') for line in table
            ]
            below.append([*doc, *commented(lines)])
        else:
            keys.append([*doc, setting(field.name, value, field.type)])

    heading = [*comment(docs[""]), f"[{name}]"] if name else []
    # a blank line before each documented key, so its comment reads as its own
    own = heading + [
        line
        for i, key in enumerate(keys)
        for line in ([""] if i and len(key) > 1 else []) + key
    ]
    return [own, *below] if own else below


def render() -> str:
    return (
        "\n\n".join(["\n".join(table) for table in [[HEADER], *tables(Config())]])
        + "\n"
    )


def render_commented(skip: set[str]) -> str:
    """Every table of the example but those in `skip`, commented out, to put
    below settings that are written. The settings at the top level are left out,
    since they cannot follow a table."""
    settings = Config()
    blocks = (
        commented(table)
        for field in fields(settings)
        if field.name not in skip
        and is_dataclass(value := getattr(settings, field.name))
        for table in tables(value, field.name)
    )
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def flat(section: object, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every setting as `(dotted.key, value)`."""
    items = (
        section.items()
        if isinstance(section, dict)
        else ((f.name, getattr(section, f.name)) for f in fields(section))  # ty: ignore[invalid-argument-type]
    )
    for name, value in items:
        if is_dataclass(value) or isinstance(value, dict):
            yield from flat(value, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}", value


def differences(path: Path) -> list[str]:
    """How the settings in `path` differ from the defaults, read the way
    `glashelder.toml` is read, so formatting does not count."""
    try:
        written = tomllib.loads(path.read_text())
        found = dict(flat(build(Config, written, prefix="")))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        return [str(error)]
    expected = dict(flat(Config()))
    # loading fills in a left-out setting with its default, so look for those first
    present = {key for key, _ in flat(written)}
    missing = [
        f"{key} is missing"
        for key, value in sorted(expected.items())
        if value is not None and key not in present
    ]
    return missing + [
        f"{key} is {toml_value(found.get(key))} instead of "
        f"{'unset' if expected.get(key) is None else toml_value(expected[key])}"
        for key in sorted(expected.keys() | found.keys())
        if found.get(key) != expected.get(key)
    ]


@click.group()
def config() -> None:
    """Show the settings glashelder.toml can hold."""


@config.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Where to write the example, by default {EXAMPLE_NAME} in the project root",
)
@click.option(
    "--check",
    is_flag=True,
    help="Write nothing, fail if the example's values differ from the defaults",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Like --check, but write the example again when it fails",
)
def example(output: Path | None, check: bool, fix: bool) -> None:
    """Write every setting with its default to glashelder.example.toml."""
    from .fs import atomic_write

    output = output or PROJECT_ROOT / EXAMPLE_NAME
    if not (check or fix):
        atomic_write(output, render())
        click.echo(f"wrote {output}")
        return
    if found := differences(output):
        click.echo(
            click.style(
                f"{output} does not match the defaults in common.config:", fg="red"
            ),
            err=True,
            color=True,
        )
        click.echo("\n".join(f"  {line}" for line in found), err=True)
        if fix and not check:
            atomic_write(output, render())
            click.echo(f"wrote {output} again", err=True)
        else:
            click.echo(
                "run `uv run glashelder config example` to write it again", err=True
            )
        sys.exit(1)

    click.echo(
        click.style("example config file matches config options", fg="green"),
        color=True,
    )


if __name__ == "__main__":
    # without the glashelder CLI, so CI needs only this package's dependencies
    config()
