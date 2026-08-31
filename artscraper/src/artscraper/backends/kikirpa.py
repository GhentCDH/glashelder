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
from . import Context, sitemap_locs, stream_sitemaps

BASE_URL = "https://balat.kikirpa.be"
SITEMAP_INDEX = f"{BASE_URL}/sitemap.xml"
OBJECT_SITEMAP = re.compile(r"sitemap-object-\d+\.xml$")
IIIF_SIZE = re.compile(r"(/full/)![^/]+(/)")
IMAGE_SIZE = 1000

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
    )


class Kikirpa:
    name = "kikirpa"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        sitemaps = [
            loc
            for loc in await sitemap_locs(ctx, SITEMAP_INDEX)
            if OBJECT_SITEMAP.search(loc)
        ]
        logger.info("kikirpa: %d object sitemaps", len(sitemaps))

        async def fetch_one(ctx: Context, url: str) -> Artwork | None:
            html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)
            return parse_page(url, html)

        def bare(url: str) -> Artwork:
            # the sitemaps carry no metadata at all, so this is url-only
            return Artwork(source="kikirpa", url=url, image_url=None)

        async for artwork in stream_sitemaps(ctx, sitemaps, fetch_one, bare):
            yield artwork
