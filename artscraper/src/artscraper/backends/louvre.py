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
from . import Backend, Context, bare, urls_to_artworks
from .parse import first_text, text

BASE_URL = "https://collections.louvre.fr"

EXPORT_CSV_ENV = "LOUVRE_EXPORT_CSV"
EXPORT_DIR_ENV = "LOUVRE_EXPORT_DIR"
DEFAULT_EXPORT_DIR: str | None = None
"""Folder of export CSVs used when neither environment variable is set."""

logger = logging.getLogger(__name__)


def export_csv_paths() -> list[Path]:
    """Return configured export CSV files in stable order."""
    if configured_csv := os.environ.get(EXPORT_CSV_ENV):
        return [Path(configured_csv)]

    export_dir = os.environ.get(EXPORT_DIR_ENV, DEFAULT_EXPORT_DIR)
    if not export_dir:
        raise ValueError(
            f"louvre: no export CSVs configured. Set {EXPORT_DIR_ENV} to the "
            f"folder with the Louvre export CSV files, or {EXPORT_CSV_ENV} to one CSV file."
        )
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
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.warning(
            "louvre: could not decode %s as UTF-8 or cp1252; falling back to latin-1",
            path,
        )
        decoded = raw.decode("latin-1")

    # Normalize Windows, old Mac, and Unix line endings.
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


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

    return [
        ark for row in reader if (ark := text(row.get("ARK") or row.get("\ufeffARK")))
    ]


def full_year(data: dict) -> str | None:
    """Return Louvre's complete human-readable creation date."""
    dates = data.get("dateCreated")

    if isinstance(dates, dict):
        dates = [dates]

    if isinstance(dates, str):
        return text(dates)

    if not isinstance(dates, list):
        return None

    parts: list[str] = []

    for date in dates:
        if found := text(date):
            parts.append(found)
            continue

        if not isinstance(date, dict):
            continue

        if found := text(date.get("text")):
            parts.append(found)
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

    return first_text(
        creator.get("label") or creator.get("name") or creator.get("title")
        if isinstance(creator, dict)
        else creator
        for creator in creators
    )


def first_image(data: dict) -> str | None:
    """Return the first available Louvre image URL."""
    images = data.get("image")

    if isinstance(images, dict):
        images = [images]

    if not isinstance(images, list):
        return None

    image_url = first_text(
        image.get("urlImage")
        or image.get("url")
        or image.get("contentUrl")
        or image.get("urlThumbnail")
        if isinstance(image, dict)
        else image
        for image in images
    )
    return urljoin(BASE_URL, image_url) if image_url else None


def record_url(data: dict, fallback_url: str) -> str:
    """Extract the record URL from Louvre JSON."""
    url = first_text(map(data.get, ("url", "@id", "sameAs")))
    return urljoin(BASE_URL, url) if url else fallback_url


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

    return Artwork(
        source="louvre",
        url=record_url(data, fallback_url),
        image_url=first_image(data),
        title=text(data.get("title")),
        artist=first_creator(data),
        year=full_year(data),
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
                f"the {EXPORT_DIR_ENV} folder, or set {EXPORT_CSV_ENV} to one CSV file."
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

        urls = [url for url in map(artwork_url, arks) if url not in ctx.seen]
        logger.info("louvre: %d unseen records from export", len(urls))

        async def fetch_one(ctx: Context, url: str) -> Artwork | None:
            body = await fetch_text(ctx.client, json_url(url), ctx.throttle, ctx.stats)
            return parse_page(url, body)

        async for artwork in urls_to_artworks(ctx, urls, fetch_one, bare(self.name)):
            yield artwork
