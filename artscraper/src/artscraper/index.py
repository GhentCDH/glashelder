"""Looking artworks back up by the reference an embedding or a match refers to."""

import json
from pathlib import Path

from .model import Artwork


def read(path: str | Path) -> list[Artwork]:
    """Read artworks from a scrape, accepting either the JSONL or the JSON list."""
    text = Path(path).read_text()
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
