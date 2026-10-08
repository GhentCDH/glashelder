"""National Gallery of Art Open Data backend.

NGA's public website may return HTTP 403 to automated requests. This backend
therefore uses NGA's official CC0 Open Data CSV files from the NGA GitHub
repository rather than scraping the public website.

The dataset provides:
- objects.csv: object metadata and classifications
- constituents.csv: artist/creator information
- published_images.csv: IIIF image identifiers and thumbnail URLs
"""

import csv
import io
import logging
import re
import unicodedata
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from ..client import fetch_text
from ..model import Artwork
from . import Backend, Context

OPEN_DATA_BASE = (
    "https://raw.githubusercontent.com/"
    "NationalGalleryOfArt/opendata/refs/heads/main/data"
)

OBJECTS_CSV = f"{OPEN_DATA_BASE}/objects.csv"
CONSTITUENTS_CSV = f"{OPEN_DATA_BASE}/constituents.csv"
PUBLISHED_IMAGES_CSV = f"{OPEN_DATA_BASE}/published_images.csv"

NGA_ARTWORK_BASE = "https://www.nga.gov/artworks"
IIIF_BASE = "https://api.nga.gov/iiif/"
IIIF_SIZE = "!1000,1000"

# Filters requested through NGA's website:
# begin_year=-499
# end_year=1935
BEGIN_YEAR = -499
END_YEAR = 1935

# These are intentionally human-readable dataset classifications, rather than
# website facet IDs. Run once with the diagnostic logging enabled below, then
# replace these with the exact classifications corresponding to the three
# requested NGA website awtype facets.
SELECTED_CLASSIFICATIONS = {
    "painting",
    "sculpture",
    "print",
}

logger = logging.getLogger(__name__)


def csv_rows(text: str) -> list[dict[str, str]]:
    """Parse an NGA Open Data UTF-8 CSV file."""
    return list(
        csv.DictReader(
            io.StringIO(text.lstrip("\ufeff")),
        )
    )


def value(row: dict[str, str], *names: str) -> str | None:
    """Return the first non-empty value among possible CSV field names."""
    for name in names:
        item = row.get(name)

        if isinstance(item, str) and item.strip():
            return item.strip()

    return None


def int_value(row: dict[str, str], *names: str) -> int | None:
    """Read an optional integer CSV field."""
    raw = value(row, *names)

    if raw is None:
        return None

    try:
        return int(raw)
    except ValueError:
        return None


def artist_names(
    object_row: dict[str, str],
    constituents: dict[str, str],
) -> str | None:
    """Return the credited artist or maker for an NGA object."""
    # NGA objects.csv already provides the museum's attribution string.
    # It is the most complete and human-readable artist/maker label.
    attribution = value(
        object_row,
        "attribution",
        "attributioninverted",
    )

    if attribution is not None:
        return attribution

    # Fallback for records without a direct attribution.
    raw_ids = value(
        object_row,
        "constituentid",
        "constituentids",
        "constituent_id",
        "constituent_ids",
    )

    if raw_ids is None:
        return None

    names: list[str] = []

    for constituent_id in raw_ids.split("|"):
        constituent_id = constituent_id.strip()

        if not constituent_id:
            continue

        name = constituents.get(constituent_id)

        if name and name not in names:
            names.append(name)

    return "; ".join(names) or None


def iiif_image_url(image_row: dict[str, str]) -> str | None:
    """Return a stable 1000px NGA IIIF JPEG URL for one published image."""
    # NGA's published_images.csv generally provides the IIIF thumbnail URL.
    # It may also provide the IIIF UUID directly, depending on data version.
    source_url = value(
        image_row,
        "iiifthumburl",
        "iiif_thumb_url",
        "iiifurl",
        "iiif_url",
    )

    if source_url:
        parts = urlsplit(source_url)

        if "/iiif/" in parts.path:
            identifier = parts.path.split("/iiif/", 1)[1].split("/full/", 1)[0]
            identifier = identifier.strip("/")

            if identifier:
                return f"{IIIF_BASE}{identifier}/full/{IIIF_SIZE}/0/default.jpg"

    uuid = value(
        image_row,
        "uuid",
        "imageuuid",
        "image_uuid",
    )

    if uuid:
        return f"{IIIF_BASE}{uuid}/full/{IIIF_SIZE}/0/default.jpg"

    return None


def image_object_id(row: dict[str, str]) -> str | None:
    """Find the object ID represented by a published-image CSV row."""
    return value(
        row,
        "depictstmsobjectid",
        "objectid",
        "object_id",
    )


def date_text(row: dict[str, str]) -> str | None:
    """Preserve NGA's complete human-readable date field."""
    return value(
        row,
        "displaydate",
        "display_date",
        "dated",
    )


def classification(row: dict[str, str]) -> str | None:
    """Read NGA's human-readable artwork classification."""
    return value(
        row,
        "visualbrowserclassification",
        "visual_browser_classification",
        "classification",
    )


def wanted_object(row: dict[str, str]) -> bool:
    """Apply the requested date range and artwork-type classifications."""
    begin_year = int_value(row, "beginyear", "begin_year")

    if begin_year is None:
        return False

    if not BEGIN_YEAR <= begin_year <= END_YEAR:
        return False

    object_classification = classification(row)

    return (
        object_classification is not None
        and object_classification.casefold() in SELECTED_CLASSIFICATIONS
    )


# Words the NGA website drops from artwork URL slugs.
# Verified by testing live URLs: articles and common prepositions are
# removed, but conjunctions like "and" are kept.
_NGA_STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "for",
    "as",
    "but",
    "or",
    "nor",
}


def nga_slug(text: str) -> str:
    """Convert an NGA artwork title to the public website URL slug."""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    slug = ascii_text.lower()

    # Turn apostrophes into nothing so "St. Mary's" becomes "st-marys".
    slug = slug.replace("'", "")

    # Remove stop words (articles, prepositions) — the NGA website drops
    # these from its URL slugs.  "and" is intentionally kept.
    words = slug.split()
    words = [w for w in words if w not in _NGA_STOP_WORDS]
    slug = " ".join(words)

    # Every run of non-alphanumeric characters becomes one hyphen.
    slug = re.sub(r"[^a-z0-9]+", "-", slug)

    return slug.strip("-")


def object_url(object_id: str, title: str) -> str:
    """Build NGA's public artwork-page URL."""
    return f"https://www.nga.gov/artworks/{object_id}-{nga_slug(title)}"


class Nga(Backend):
    """Yield NGA Open Data objects with published NGA IIIF images."""

    name = "nga"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        """Yield selected NGA objects with direct NGA IIIF image URLs."""
        logger.info("nga: downloading official Open Data CSV files")

        objects_text = await fetch_text(
            ctx.client,
            OBJECTS_CSV,
            ctx.throttle,
            ctx.stats,
        )
        constituents_text = await fetch_text(
            ctx.client,
            CONSTITUENTS_CSV,
            ctx.throttle,
            ctx.stats,
        )
        images_text = await fetch_text(
            ctx.client,
            PUBLISHED_IMAGES_CSV,
            ctx.throttle,
            ctx.stats,
        )

        object_rows = csv_rows(objects_text)
        constituent_rows = csv_rows(constituents_text)
        image_rows = csv_rows(images_text)

        constituents: dict[str, str] = {}

        for row in constituent_rows:
            constituent_id = value(
                row,
                "constituentid",
                "constituent_id",
                "id",
            )
            name = value(
                row,
                "preferreddisplayname",
                "preferredname",
                "preferred_name",
                "forwarddisplayname",
                "displayname",
                "display_name",
                "name",
            )

            if constituent_id and name:
                constituents[constituent_id] = name

        images: dict[str, str] = {}

        for row in image_rows:
            object_id = image_object_id(row)
            image_url = iiif_image_url(row)

            if object_id and image_url and object_id not in images:
                images[object_id] = image_url

        logger.info(
            "nga: %d objects, %d constituents, %d objects with IIIF images",
            len(object_rows),
            len(constituents),
            len(images),
        )

        yielded = 0

        for row in object_rows:
            if not wanted_object(row):
                continue

            object_id = value(
                row,
                "objectid",
                "object_id",
            )

            if object_id is None:
                continue

            image_url = images.get(object_id)

            # Glashelder needs a usable image URL, so skip NGA records whose
            # Open Data entry has no published IIIF image.
            if image_url is None:
                continue

            title = value(
                row,
                "title",
                "objecttitle",
                "object_title",
            )

            if title is None:
                continue

            url = object_url(object_id, title)

            if url in ctx.seen:
                continue

            yield Artwork(
                source="nga",
                url=url,
                image_url=image_url,
                title=title,
                artist=artist_names(row, constituents),
                year=date_text(row),
            )

            yielded += 1

            if ctx.limit and yielded >= ctx.limit:
                return
