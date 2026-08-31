"""The unified artwork record every backend yields."""

from dataclasses import dataclass


@dataclass(slots=True)
class Artwork:
    """One artwork, as described by an external data source."""

    source: str
    """Name of the backend this came from."""
    url: str
    """Page where more information about the artwork can be found."""
    image_url: str | None
    """Direct link to an image, if the backend could find one."""
    title: str | None = None
    artist: str | None = None
    year: str | None = None
    """Free-form: sources give things like "1401/1500" or "c. 1650"."""
