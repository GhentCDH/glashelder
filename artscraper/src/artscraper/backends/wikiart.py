"""wikiart.org.

The sitemaps list every artwork page directly, which saves crawling the
alphabet and artist listings the way `glasgetter` had to.
"""

import logging
import re
from collections.abc import AsyncIterator

from bs4 import BeautifulSoup, Tag

from ..client import RestrictedError, fetch_text
from ..model import Artwork
from . import Backend, Context, stream_sitemaps

BASE_URL = "https://www.wikiart.org"
PAINTING_SITEMAPS = [f"{BASE_URL}/sitemap/paintings-{n}.xml" for n in range(1, 7)]
ARTWORK_PATH = re.compile(r"^/en/([^/]+)/([^/]+)$")
TRAILING_YEAR = re.compile(r"-(\d{4})$")

logger = logging.getLogger(__name__)


def meta_content(soup: BeautifulSoup, prop: str) -> str | None:
    """The content of an Open Graph meta tag, if it has one."""
    tag = soup.find("meta", property=prop)
    content = tag.get("content") if isinstance(tag, Tag) else None
    return content if isinstance(content, str) and content else None


def slug_to_name(slug: str) -> str:
    return slug.replace("-", " ").strip().title()


def parse_artwork_url(url: str) -> Artwork | None:
    """Read artist, title and year straight out of an artwork URL.

    The sitemaps also hold listing pages such as `/en/popular-paintings/alltime`,
    which look the same but are not artworks; those are filtered out later by the
    page fetch, so only the obvious shape is checked here.
    """
    match = ARTWORK_PATH.match(url.removeprefix(BASE_URL))
    if match is None:
        return None

    artist_slug, work_slug = match.groups()
    year = None
    if (found := TRAILING_YEAR.search(work_slug)) and 1000 <= int(found[1]) <= 2099:
        year = found[1]
        work_slug = work_slug[: found.start()]

    return Artwork(
        source="wikiart",
        url=url,
        image_url=None,
        title=slug_to_name(work_slug),
        artist=slug_to_name(artist_slug),
        year=year,
    )


def parse_image_url(html: str) -> str | None:
    """Pull the artwork image out of a page's Open Graph tags.

    Raises `RestrictedError` for artworks wikiart will not show.
    """
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one(".restricted-msg") is not None:
        raise RestrictedError

    src = meta_content(soup, "og:image")
    if src is None:
        return None

    # strip the image processing suffix, e.g. "...jpg!Large.jpg"
    return src.split("!", 1)[0]


def parse_page(url: str, html: str) -> Artwork | None:
    artwork = parse_artwork_url(url)
    if artwork is None:
        return None

    if 'itemtype="https://schema.org/Painting"' not in html:
        # a listing page such as /en/popular-paintings/alltime, which has the
        # same url shape as an artwork
        return None

    artwork.image_url = parse_image_url(html)

    if title := meta_content(BeautifulSoup(html, "html.parser"), "og:title"):
        # format: "Autumn - Giuseppe Arcimboldo - WikiArt.org"
        artwork.title = title.split(" - ")[0].strip()

    return artwork


class WikiArt(Backend):
    name = "wikiart"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        async def fetch_one(ctx: Context, url: str) -> Artwork | None:
            html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)
            return parse_page(url, html)

        async for artwork in stream_sitemaps(
            ctx, PAINTING_SITEMAPS, fetch_one, parse_artwork_url
        ):
            yield artwork
