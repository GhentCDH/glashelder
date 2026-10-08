"""Reading scrapes back, and looking artworks up by the reference an embedding or a match refers to."""

import json
from pathlib import Path
from typing import IO

import zstandard as zstd

from .model import Artwork


def open_text(path: str | Path, mode: str = "r") -> IO[str]:
    """Open a text file, (de)compressing on the fly when it ends in `.zst`."""
    path = Path(path)
    if path.suffix.lower() == ".zst":
        return zstd.open(path, f"{mode}t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read(path: str | Path) -> list[Artwork]:
    """Read artworks from a scrape: the JSONL or the JSON list, optionally zstd compressed."""
    with open_text(path) as file:
        text = file.read()
    if text.lstrip().startswith("["):
        return [Artwork(**row) for row in json.loads(text)]
    return [Artwork(**json.loads(line)) for line in text.splitlines() if line.strip()]


def index(path: str | Path) -> dict[str, Artwork]:
    """Artworks keyed by every reference that can point at them.

    Both the image URL and the description URL are keys, so whatever a downstream
    result carries, the record is one lookup away. Nothing else needs to know how
    the scrape was stored.
    """
    found: dict[str, Artwork] = {}
    for artwork in read(path):
        found[artwork.url] = artwork
        if artwork.image_url:
            found[artwork.image_url] = artwork
    return found


def artwork_of(reference: str, artworks: dict[str, Artwork] | None) -> Artwork | None:
    """The scraped record a matched reference describes, if we have it.

    The reference is the join key: a URL for scraped artworks, a path for local
    files (looked up by name, since the directory may differ per machine).
    """
    if not artworks or not reference:
        return None
    return artworks.get(reference) or artworks.get(Path(reference).name)
