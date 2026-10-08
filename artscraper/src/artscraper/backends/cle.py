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
from . import Backend, Context, enrich_pool

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


def text_value(value: object) -> str | None:
    """Return a cleaned string from a metadata value."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def first_text(data: dict, *fields: str) -> str | None:
    """Return the first non-empty text value."""
    for field in fields:
        value = text_value(data.get(field))

        if value:
            return value

    return None


def first_creator(data: dict) -> str | None:
    """Extract the first creator or artist value."""
    for field in ("creator", "artist", "author"):
        value = data.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, list):
            creators = [
                item.strip() for item in value if isinstance(item, str) and item.strip()
            ]

            if creators:
                return "; ".join(creators)

    return None


def first_year(data: dict) -> str | None:
    """Extract the original date or year."""
    for field in ("date", "year", "created"):
        value = data.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, int):
            return str(value)

    return None


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

        name = file.get("name")

        if not isinstance(name, str) or not name.strip():
            continue

        name = name.strip()
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

    identifiers: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        identifier = item.get("identifier")

        if isinstance(identifier, str) and identifier.strip():
            identifiers.append(identifier.strip())

    cursor = data.get("cursor")

    if not isinstance(cursor, str) or not cursor.strip():
        cursor = None

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

    title = first_text(
        metadata,
        "title",
        "name",
    )

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
        yielded = 0
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
                artwork_url(identifier)
                for identifier in new_identifiers
                if artwork_url(identifier) not in ctx.seen
            ]

            if ctx.with_images:
                remaining = ctx.limit - yielded if ctx.limit else None

                async def fetch_one(
                    ctx: Context,
                    url: str,
                ) -> Artwork | None:
                    identifier = url.rstrip("/").rsplit("/", 1)[-1]

                    body = await fetch_text(
                        ctx.client,
                        metadata_url(identifier),
                        ctx.throttle,
                        ctx.stats,
                    )

                    return parse_metadata(
                        identifier,
                        body,
                    )

                async for artwork in enrich_pool(
                    ctx,
                    urls,
                    fetch_one,
                    remaining,
                ):
                    yield artwork
                    yielded += 1

                    if ctx.limit and yielded >= ctx.limit:
                        return

            else:
                for url in urls:
                    yield Artwork(
                        source="cleveland",
                        url=url,
                        image_url=None,
                    )

                    yielded += 1

                    if ctx.limit and yielded >= ctx.limit:
                        return

            # The scrape API returns no cursor once the result set is
            # exhausted. This -- not batch size or dedup counts -- is the
            # authoritative signal that there is nothing left to fetch.
            if not next_cursor:
                return

            cursor = next_cursor
