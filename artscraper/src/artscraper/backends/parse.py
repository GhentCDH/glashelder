"""Small parsing helpers the backends share."""

import json
from collections.abc import Iterable
from typing import Any

from bs4 import BeautifulSoup

# a tuple, not a set: `@type` may be an unhashable list
ARTWORK_TYPES = ("VisualArtwork", "CreativeWork", "Artwork")


def text(value: object) -> str | None:
    """`value` stripped, or None when it is not a string or only whitespace."""
    return (value.strip() or None) if isinstance(value, str) else None


def first_text(values: Iterable[object]) -> str | None:
    """The first of `values` that `text` accepts."""
    return next(filter(None, map(text, values)), None)


def parse_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Every schema.org JSON-LD object on a page, with `@graph`s and lists flattened."""
    result: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            graph = value.get("@graph")
            if isinstance(graph, list):
                add(graph)
            else:
                result.append(value)

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            add(json.loads(tag.string or tag.get_text()))
        except json.JSONDecodeError:
            continue

    return result


def artwork_json_ld(soup: BeautifulSoup) -> dict[str, Any] | None:
    """The JSON-LD object on a page that looks most like the artwork."""
    data = parse_json_ld(soup)
    return next(
        (
            item
            for item in data
            if item.get("@type") in ARTWORK_TYPES
            or (item.get("name") and (item.get("image") or item.get("creator")))
        ),
        data[0] if data else None,
    )
