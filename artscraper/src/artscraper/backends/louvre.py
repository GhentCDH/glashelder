import asyncio
import csv
import io
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote, urljoin

from ..client import fetch_text
from ..model import Artwork
from . import Backend, Context, complete_pages, enrich_pool, physical

BASE_URL = "https://collections.louvre.fr"

DEFAULT_EXPORT_CSV = "louvre_export.csv"
DEFAULT_EXPORT_DIR = "/Volumes/LG-JP-BCKUP/hannah/louvre_exports"

logger = logging.getLogger(__name__)


def export_csv_paths() -> list[Path]:
    """Return configured export CSV files in stable order."""
    configured_csv = os.environ.get("LOUVRE_EXPORT_CSV")

    if configured_csv:
        return [Path(configured_csv)]

    export_dir = os.environ.get("LOUVRE_EXPORT_DIR", DEFAULT_EXPORT_DIR)
    return sorted(Path(export_dir).glob("*.csv"))


def artwork_url(ark: str) -> str:
    """Convert an ARK identifier into a Louvre collection URL."""
    ark = ark.strip()

    # Avoid duplicating the ARK prefix when the CSV already contains it.
    if ark.startswith("ark:/53355/"):
        ark = ark.removeprefix("ark:/53355/")

    return f"{BASE_URL}/en/ark:/53355/{quote(ark, safe='')}"


def json_url(url: str) -> str:
    """Return the Louvre JSON endpoint for an artwork URL."""
    return f"{url.rstrip('/')}.json"


def read_export_text(path: Path) -> str:
    """Read a Louvre CSV while tolerating encodings and line endings."""
    raw = path.read_bytes()

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.warning(
            "louvre: could not decode %s as UTF-8 or cp1252; falling back to latin-1",
            path,
        )
        text = raw.decode("latin-1")

    # Normalize Windows, old Mac, and Unix line endings.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_export(csv_text: str) -> list[str]:
    """Extract ARK identifiers from a Louvre CSV export."""
    reader = csv.DictReader(
        io.StringIO(csv_text.lstrip("\ufeff"), newline=""),
        delimiter=";",
    )

    if reader.fieldnames is None:
        logger.warning("louvre: CSV has no header row")
        return []

    logger.info("louvre: CSV columns: %s", reader.fieldnames)

    arks: list[str] = []

    for row in reader:
        ark = row.get("ARK") or row.get("\ufeffARK")

        if isinstance(ark, str) and ark.strip():
            arks.append(ark.strip())

    return arks


def full_year(data: dict) -> str | None:
    """Return Louvre's complete human-readable creation date."""
    dates = data.get("dateCreated")

    if isinstance(dates, dict):
        dates = [dates]

    if isinstance(dates, str) and dates.strip():
        return dates.strip()

    if not isinstance(dates, list):
        return None

    parts: list[str] = []

    for date in dates:
        if isinstance(date, str) and date.strip():
            parts.append(date.strip())
            continue

        if not isinstance(date, dict):
            continue

        text = date.get("text")

        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue

        start_year = date.get("startYear")
        end_year = date.get("endYear")

        if isinstance(start_year, int) and isinstance(end_year, int):
            if start_year == end_year:
                parts.append(str(start_year))
            else:
                parts.append(f"{start_year} / {end_year}")
        elif isinstance(start_year, int):
            parts.append(str(start_year))
        elif isinstance(end_year, int):
            parts.append(str(end_year))

    return " ; ".join(parts) or None


def first_creator(data: dict) -> str | None:
    """Return the first available creator label."""
    creators = data.get("creator")

    if isinstance(creators, dict):
        creators = [creators]

    if not isinstance(creators, list):
        return None

    for creator in creators:
        if isinstance(creator, str) and creator.strip():
            return creator.strip()

        if not isinstance(creator, dict):
            continue

        label = creator.get("label") or creator.get("name") or creator.get("title")

        if isinstance(label, str) and label.strip():
            return label.strip()

    return None


def first_image(data: dict) -> str | None:
    """Return the first available Louvre image URL."""
    images = data.get("image")

    if isinstance(images, dict):
        images = [images]

    if not isinstance(images, list):
        return None

    for image in images:
        if isinstance(image, str) and image.strip():
            return urljoin(BASE_URL, image.strip())

        if not isinstance(image, dict):
            continue

        image_url = (
            image.get("urlImage")
            or image.get("url")
            or image.get("contentUrl")
            or image.get("urlThumbnail")
        )

        if isinstance(image_url, str) and image_url.strip():
            return urljoin(BASE_URL, image_url.strip())

    return None


def record_url(data: dict, fallback_url: str) -> str:
    """Extract the record URL from Louvre JSON."""
    for key in ("url", "@id", "sameAs"):
        value = data.get(key)

        if isinstance(value, str) and value.strip():
            return urljoin(BASE_URL, value.strip())

    return fallback_url


def physical_description(data: dict) -> str | None:
    """Object type, materials and techniques, and size, as the Louvre gives them (in French)."""
    denominations = [
        entry.get("value")
        for entry in data.get("denominationTitle") or []
        if isinstance(entry, dict) and entry.get("type") == "Dénomination"
    ]
    materials = data.get("materialsAndTechniques")
    dimensions = data.get("dimension")
    size = (
        ", ".join(
            f"{entry.get('type')} {entry.get('displayDimension')}"
            for entry in dimensions
            if isinstance(entry, dict) and entry.get("displayDimension")
        )
        if isinstance(dimensions, list)
        else dimensions
    )
    return physical(
        data.get("objectType"),
        *denominations,
        ", ".join(materials.splitlines()) if isinstance(materials, str) else None,
        size,
    )


async def fetch_record(ctx: Context, url: str) -> Artwork | None:
    """Fetch and parse the JSON record behind an artwork page."""
    try:
        body = await fetch_text(ctx.client, json_url(url), ctx.throttle, ctx.stats)
    except Exception:
        logger.exception("louvre: failed to fetch %s", json_url(url))
        return None
    return parse_page(url, body)


def parse_page(fallback_url: str, body: str) -> Artwork | None:
    """Convert one Louvre JSON response into an Artwork."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("louvre: invalid JSON at %s", fallback_url)
        return None

    if not isinstance(data, dict):
        logger.warning("louvre: unexpected JSON structure at %s", fallback_url)
        return None

    title = data.get("title")

    if not isinstance(title, str):
        title = None
    else:
        title = title.strip() or None

    return Artwork(
        source="louvre",
        url=record_url(data, fallback_url),
        image_url=first_image(data),
        title=title,
        artist=first_creator(data),
        year=full_year(data),
        physical_description=physical_description(data),
    )


class Louvre(Backend):
    """Read ARK IDs from Louvre CSV exports and enrich them via JSON."""

    name = "louvre"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        paths = [path for path in export_csv_paths() if path.is_file()]

        if not paths:
            raise FileNotFoundError(
                "louvre: no export CSV files found. Export the Louvre "
                "collection pages manually and save the CSV files into "
                f"{os.environ.get('LOUVRE_EXPORT_DIR', DEFAULT_EXPORT_DIR)}, "
                "or set LOUVRE_EXPORT_CSV to one CSV file."
            )

        seen_arks: set[str] = set()
        arks: list[str] = []

        for path in paths:
            csv_text = await asyncio.to_thread(read_export_text, path)
            page_arks = parse_export(csv_text)

            new_arks = [ark for ark in page_arks if ark not in seen_arks]

            seen_arks.update(new_arks)
            arks.extend(new_arks)

            logger.info(
                "louvre: %d ARKs found, %d new, from %s",
                len(page_arks),
                len(new_arks),
                path,
            )

        logger.info(
            "louvre: %d unique ARKs found in %d file(s)",
            len(arks),
            len(paths),
        )

        urls: list[str] = []

        for ark in arks:
            url = artwork_url(ark)

            if url not in ctx.seen:
                urls.append(url)

        logger.info(
            "louvre: %d unseen records from export",
            len(urls),
        )

        if not urls:
            return

        yielded = 0

        if not ctx.with_images:
            for url in urls:
                yield Artwork(
                    source="louvre",
                    url=url,
                    image_url=None,
                )

                yielded += 1

                if ctx.limit and yielded >= ctx.limit:
                    return

            return

        async for artwork in enrich_pool(ctx, urls, fetch_record):
            if artwork is None:
                continue

            yield artwork

            yielded += 1

            if ctx.limit and yielded >= ctx.limit:
                return

    async def complete(
        self, ctx: Context, artworks: list[Artwork]
    ) -> AsyncIterator[Artwork]:
        async for artwork in complete_pages(ctx, artworks, fetch_record):
            yield artwork
