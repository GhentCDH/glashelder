"""Cleveland Museum of Art backend.

Discovers image items from the Internet Archive's ``clevelandart``
collection using the scrape API (stable, cursor-based pagination), then
fetches each item's metadata to extract the title, creator, date, and
image URL.
"""

import json
import logging
from collections.abc import AsyncIterator
from urllib.parse import quote, urlencode

from ..client import fetch_text
from ..model import Artwork
from . import Backend, Context, bare, urls_to_artworks
from .parse import first_text, text

BASE_URL = "https://archive.org"
COLLECTION = "clevelandart"

SCRAPE_URL = f"{BASE_URL}/services/search/v1/scrape"
METADATA_URL = f"{BASE_URL}/metadata"

PAGE_SIZE = 1_000

logger = logging.getLogger(__name__)


def scrape_url(cursor: str | None) -> str:
    """Build one Internet Archive scrape API URL."""
    params = {
        "q": f"collection:{COLLECTION} AND mediatype:image",
        "fields": "identifier",
        "count": str(PAGE_SIZE),
    }

    if cursor:
        params["cursor"] = cursor

    return f"{SCRAPE_URL}?{urlencode(params)}"


def metadata_url(identifier: str) -> str:
    """Build the Internet Archive item metadata URL."""
    return f"{METADATA_URL}/{quote(identifier, safe='')}"


def artwork_url(identifier: str) -> str:
    """Build the public Internet Archive item URL."""
    return f"{BASE_URL}/details/{quote(identifier, safe='')}"


def creator_text(value: object) -> str | None:
    """A creator field as text, joining lists of names."""
    if isinstance(value, list):
        return "; ".join(filter(None, map(text, value))) or None
    return text(value)


def year_text(value: object) -> str | None:
    """A date field as text, accepting bare integer years."""
    return str(value) if isinstance(value, int) else text(value)


def first_creator(data: dict) -> str | None:
    """Extract the first creator or artist value."""
    return next(
        filter(
            None, (creator_text(data.get(f)) for f in ("creator", "artist", "author"))
        ),
        None,
    )


def first_year(data: dict) -> str | None:
    """Extract the original date or year."""
    return next(
        filter(None, (year_text(data.get(f)) for f in ("date", "year", "created"))),
        None,
    )


def image_from_files(
    files: object,
    identifier: str,
) -> str | None:
    """Return the first suitable public image file URL."""
    if not isinstance(files, list):
        return None

    preferred_extensions = ".jpg"

    candidates: list[tuple[int, str]] = []

    for file in files:
        if not isinstance(file, dict):
            continue

        name = text(file.get("name"))

        if name is None:
            continue

        lower_name = name.casefold()

        # Ignore thumbnails, derivatives, metadata, and archives.
        if any(
            token in lower_name
            for token in (
                "_thumb",
                "thumb",
                "thumbnail",
                "original_jp2",
                "itemimage",
                ".xml",
                ".json",
                ".txt",
                ".pdf",
                ".zip",
                ".tar",
                ".gz",
            )
        ):
            continue

        if not lower_name.endswith(preferred_extensions):
            continue

        priority = 0

        if "original" in lower_name:
            priority -= 20

        if "full" in lower_name:
            priority -= 10

        candidates.append((priority, name))

    if not candidates:
        return None

    _, filename = sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    )[0]

    return (
        f"{BASE_URL}/download/{quote(identifier, safe='')}/{quote(filename, safe='')}"
    )


def parse_scrape(body: str) -> tuple[list[str], str | None, int]:
    """Parse one Internet Archive scrape API response.

    Returns:
        A tuple containing:
        - identifiers found in this batch;
        - the cursor to request the next batch, or None if exhausted;
        - the total number of matching items reported by the API.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("cleveland: scrape returned invalid JSON")
        return [], None, 0

    if not isinstance(data, dict):
        return [], None, 0

    items = data.get("items", [])

    if not isinstance(items, list):
        items = []

    identifiers = [
        identifier
        for item in items
        if isinstance(item, dict) and (identifier := text(item.get("identifier")))
    ]
    cursor = text(data.get("cursor"))

    total = data.get("total", 0)
    total = int(total) if isinstance(total, int) else 0

    return identifiers, cursor, total


def parse_metadata(
    identifier: str,
    body: str,
) -> Artwork | None:
    """Create an Artwork only when the item has an image file."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.warning(
            "cleveland: invalid metadata JSON for %s",
            identifier,
        )
        return None

    if not isinstance(payload, dict):
        return None

    metadata = payload.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    image_url = image_from_files(
        payload.get("files"),
        identifier,
    )

    # Exclude items without a usable image.
    if image_url is None:
        logger.debug(
            "cleveland: skipping %s because no image file was found",
            identifier,
        )
        return None

    title = first_text(map(metadata.get, ("title", "name")))

    artist = first_creator(metadata)
    year = first_year(metadata)

    return Artwork(
        source="cleveland",
        url=artwork_url(identifier),
        image_url=image_url,
        title=title,
        artist=artist,
        year=year,
    )


class Cleveland(Backend):
    """Scrape image items from the Cleveland Museum of Art collection."""

    name = "cleveland"

    async def stream(
        self,
        ctx: Context,
    ) -> AsyncIterator[Artwork]:
        async def fetch_one(ctx: Context, url: str) -> Artwork | None:
            identifier = url.rstrip("/").rsplit("/", 1)[-1]
            body = await fetch_text(
                ctx.client, metadata_url(identifier), ctx.throttle, ctx.stats
            )
            return parse_metadata(identifier, body)

        discovered: set[str] = set()
        cursor: str | None = None

        while True:
            body = await fetch_text(
                ctx.client,
                scrape_url(cursor),
                ctx.throttle,
                ctx.stats,
            )

            identifiers, next_cursor, total = parse_scrape(body)

            new_identifiers = [
                identifier for identifier in identifiers if identifier not in discovered
            ]

            discovered.update(new_identifiers)

            logger.info(
                "cleveland: %d new identifiers this batch "
                "(raw batch: %d, discovered so far: %d, total reported: %d)",
                len(new_identifiers),
                len(identifiers),
                len(discovered),
                total,
            )

            urls = [
                url for url in map(artwork_url, new_identifiers) if url not in ctx.seen
            ]
            async for artwork in urls_to_artworks(
                ctx, urls, fetch_one, bare(self.name)
            ):
                yield artwork

            # The scrape API returns no cursor once the result set is
            # exhausted. This -- not batch size or dedup counts -- is the
            # authoritative signal that there is nothing left to fetch.
            if not next_cursor:
                return

            cursor = next_cursor
