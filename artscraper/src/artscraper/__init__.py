"""Scrape an index of externally described artworks to match against."""

from .harness import scrape
from .index import artwork_of, index, open_text, read
from .model import Artwork, merge

__all__ = ["Artwork", "artwork_of", "index", "merge", "open_text", "read", "scrape"]
