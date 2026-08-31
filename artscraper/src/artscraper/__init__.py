"""Scrape an index of externally described artworks to match against."""

from .harness import scrape
from .model import Artwork

__all__ = ["Artwork", "scrape"]
