"""Scrape an index of externally described artworks to match against."""

from .harness import scrape
from .index import index, read
from .model import Artwork

__all__ = ["Artwork", "index", "read", "scrape"]
