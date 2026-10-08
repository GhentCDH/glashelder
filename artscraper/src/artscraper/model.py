"""The unified artwork record every backend yields."""

from dataclasses import dataclass, fields


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
    physical_description: str | None = None
    """Free-form, e.g. "oil on canvas; 73 x 92 cm": whatever the source says about
    medium, support, object type and size."""


def merge(old: Artwork, new: Artwork) -> Artwork:
    """`old`, with only its missing fields taken from `new`."""
    return Artwork(
        **{
            f.name: old_value
            if (old_value := getattr(old, f.name)) is not None
            else getattr(new, f.name)
            for f in fields(Artwork)
        }
    )
