"""balat.kikirpa.be.

The site was rebuilt: the `results.php` search `glasgetter` used now answers 403
and robots.txt disallows the new search URLs. What is left is the object
sitemaps plus a schema.org JSON-LD block on every detail page.
"""

import json
import logging
import re
from collections.abc import AsyncIterator

from bs4 import BeautifulSoup

from ..client import fetch_text
from ..model import Artwork
from . import (
    Backend,
    Context,
    complete_pages,
    physical,
    sitemap_locs,
    stream_sitemaps,
)

BASE_URL = "https://balat.kikirpa.be"
SITEMAP_INDEX = f"{BASE_URL}/sitemap.xml"
OBJECT_SITEMAP = re.compile(r"sitemap-object-\d+\.xml$")
IIIF_SIZE = re.compile(r"(/full/)![^/]+(/)")
IMAGE_SIZE = 1000
THESAURUS_TAG = re.compile(r"\[[^\]]*\]")

logger = logging.getLogger(__name__)


def full_image_url(url: str) -> str:
    """Ask IIIF for a usable size instead of the page's thumbnail size."""
    return IIIF_SIZE.sub(rf"\1!{IMAGE_SIZE},{IMAGE_SIZE}\2", url)


def parse_json_ld(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", type="application/ld+json")
    if tag is None:
        return None
    try:
        data = json.loads(tag.string or "")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def texts(value: object) -> list[str]:
    """A JSON-LD value that may be one string or a list of them, without thesaurus
    tags such as "[matière]"."""
    items = (
        [value] if isinstance(value, str) else value if isinstance(value, list) else []
    )
    return [
        THESAURUS_TAG.sub("", item).strip() for item in items if isinstance(item, str)
    ]


def physical_description(data: dict) -> str | None:
    """Art form, medium and size, e.g. "Sculpture, Bois; chêne; height 252 cm"."""
    size = ", ".join(
        f"{key} {measure.get('value')} {measure.get('unitText') or ''}".strip()
        for key in ("height", "width", "depth")
        if isinstance(measure := data.get(key), dict) and measure.get("value")
    )
    return physical(
        ", ".join(texts(data.get("artform"))),
        ", ".join(texts(data.get("artMedium"))),
        size,
    )


async def fetch_page(ctx: Context, url: str) -> Artwork | None:
    html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)
    return parse_page(url, html)


def parse_page(url: str, html: str) -> Artwork | None:
    data = parse_json_ld(html)
    if data is None:
        logger.warning("no usable json-ld at %s", url)
        return None

    image = data.get("image")
    creator = data.get("creator") or {}

    return Artwork(
        source="kikirpa",
        url=data.get("url") or url,
        image_url=full_image_url(image) if isinstance(image, str) else None,
        title=data.get("name"),
        artist=creator.get("name") if isinstance(creator, dict) else None,
        year=data.get("dateCreated"),
        physical_description=physical_description(data),
    )


class Kikirpa(Backend):
    name = "kikirpa"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        sitemaps = [
            loc
            for loc in await sitemap_locs(ctx, SITEMAP_INDEX)
            if OBJECT_SITEMAP.search(loc)
        ]
        logger.info("kikirpa: %d object sitemaps", len(sitemaps))

        def bare(url: str) -> Artwork:
            # the sitemaps carry no metadata at all, so this is url-only
            return Artwork(source="kikirpa", url=url, image_url=None)

        async for artwork in stream_sitemaps(ctx, sitemaps, fetch_page, bare):
            yield artwork

    async def complete(
        self, ctx: Context, artworks: list[Artwork]
    ) -> AsyncIterator[Artwork]:
        async for artwork in complete_pages(ctx, artworks, fetch_page):
            yield artwork
