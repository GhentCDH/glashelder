"""Uffizi Galleries (uffizi.it).

The Uffizi artwork index is rendered dynamically, so the category pages are
used as an index and each artwork detail page is fetched for its image and
metadata. The three supplied collections are covered:

- Department of Prints and Drawings
- Painting
- Sculpture

The index supports ``?page=N`` pagination. We keep following pages until a
page produces no new artwork URLs.
"""

import logging
import re
from collections.abc import AsyncIterator, Iterable
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..client import fetch_text
from ..model import Artwork
from . import Backend, Context, enrich_pool, sitemap_locs
from .parse import artwork_json_ld, text

BASE_URL = "https://www.uffizi.it"
CATEGORY_INDEXES = {
    f"{BASE_URL}/en/the-uffizi/prints-and-drawings/artworks": "department of prints and drawings",
    f"{BASE_URL}/en/the-uffizi/painting/artworks": "painting",
    f"{BASE_URL}/en/the-uffizi/sculpture/artworks": "sculpture",
}
"""Category index pages and the collection each one covers."""

ARTWORK_RE = re.compile(r"^/en/artworks/(?!search/?$)[^/?#]+/?$")
logger = logging.getLogger(__name__)


def canonical_artwork_url(href: str) -> str | None:
    """The canonical form of a Uffizi artwork detail URL, None for anything else."""
    parsed = urlparse(urljoin(BASE_URL, href.strip()))
    if parsed.netloc not in {"www.uffizi.it", "uffizi.it"}:
        return None
    if not ARTWORK_RE.fullmatch(parsed.path):
        return None
    return f"{BASE_URL}{parsed.path.rstrip('/')}"


def artwork_urls(hrefs: Iterable[str]) -> list[str]:
    """The distinct artwork detail URLs among `hrefs`, in order."""
    return list(dict.fromkeys(filter(None, map(canonical_artwork_url, hrefs))))


def index_artwork_urls(html: str) -> list[str]:
    """Extract Uffizi artwork detail URLs from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    return artwork_urls(str(tag["href"]) for tag in soup.find_all("a", href=True))


async def sitemap_artwork_urls(ctx: Context) -> list[str]:
    """Fallback discovery for the current Uffizi site.

    The three /artworks index pages are client-rendered and can contain no
    artwork anchors in the HTML returned to the scraper. The site's sitemap
    is therefore used as a complete URL index when that happens.
    """
    return artwork_urls(await sitemap_locs(ctx, f"{BASE_URL}/sitemap.xml"))


def _person_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return text(value.get("name"))
    if isinstance(value, list):
        return ", ".join(filter(None, map(_person_name, value))) or None
    return text(value)


def _middle_srcset_url(srcset: str) -> str | None:
    """Return the middle-resolution URL from a srcset string."""
    candidates: list[tuple[int, str]] = []

    for item in srcset.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split()
        url = parts[0]

        width = 0
        if len(parts) > 1 and parts[1].endswith("w"):
            try:
                width = int(parts[1][:-1])
            except ValueError:
                pass

        candidates.append((width, url))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[len(candidates) // 2][1]


def _image_from_html(soup: BeautifulSoup, data: dict[str, Any] | None) -> str | None:
    """Return a single middle-resolution artwork image URL."""
    for img in soup.find_all("img"):
        srcset = img.get("srcset")
        if not isinstance(srcset, str) or not srcset.strip():
            continue

        if "datocms-assets.com" in srcset and "og-placeholder" not in srcset:
            return _middle_srcset_url(srcset)

    for source in soup.find_all("source"):
        srcset = source.get("srcset")
        if not isinstance(srcset, str) or not srcset.strip():
            continue

        if "datocms-assets.com" in srcset and "og-placeholder" not in srcset:
            return _middle_srcset_url(srcset)

    # Fallbacks
    candidates: list[Any] = []
    if data is not None:
        candidates.append(data.get("image"))

    for prop in ("og:image", "twitter:image"):
        tag = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        if tag and tag.get("content"):
            candidates.append(tag["content"])

    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("url") or candidate.get("content")
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None

        if isinstance(candidate, str) and candidate.strip():
            value = candidate.strip()
            if "og-placeholder" not in value:
                return urljoin(BASE_URL, value)

    return None


def _date_from_page(soup: BeautifulSoup) -> str | None:
    """Extract the artwork Date value from the Uffizi detail page."""
    # Look for a label/value pair in the page text.
    for element in soup.find_all(string=re.compile(r"^\s*Date\s*$", re.I)):
        parent = element.parent
        if parent is None:
            continue

        # Try the next sibling / nearby element.
        for candidate in (
            parent.find_next_sibling(),
            parent.parent.find_next_sibling() if parent.parent else None,
        ):
            if candidate:
                value = candidate.get_text(" ", strip=True)
                if value and value.lower() != "date":
                    return value

    # Fallback: search the visible text.
    text = soup.get_text("\n", strip=True)
    match = re.search(
        r"(?im)^\s*Date\s*$\s*\n\s*([^\n]+)",
        text,
    )
    if match:
        return match.group(1).strip()

    return None


def parse_page(url: str, html: str) -> Artwork | None:
    soup = BeautifulSoup(html, "html.parser")
    data = artwork_json_ld(soup)

    title = (data or {}).get("name")
    if not title:
        heading = soup.find("h1")
        title = heading.get_text(" ", strip=True) if heading else None

    artist = _person_name(
        (data or {}).get("creator")
        or (data or {}).get("artist")
        or (data or {}).get("author")
    )

    if artist is None:
        headings = soup.find_all(["h1", "h2"])
        if len(headings) > 1:
            artist = headings[1].get_text(" ", strip=True) or None

    image_url = _image_from_html(soup, data)

    if image_url is None:
        logger.warning(
            "uffizi: no image found at %s; img=%d source=%d jsonld=%s",
            url,
            len(soup.find_all("img")),
            len(soup.find_all("source")),
            bool(data),
        )
        return None

    date = _date_from_page(soup)

    return Artwork(
        source="uffizi",
        url=(data or {}).get("url") or url,
        image_url=image_url,
        title=title,
        artist=artist,
        year=date
        or (data or {}).get("dateCreated")
        or (data or {}).get("datePublished"),
    )


async def fetch_one(ctx: Context, url: str) -> Artwork | None:
    html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)
    return parse_page(url, html)


class Uffizi(Backend):
    name = "uffizi"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        # First try the supplied category index pages. On the current Uffizi
        # site these pages are client-rendered, so the HTML may contain zero
        # artwork links. In that case fall back to the site's sitemap.

        # artwork URL to its collection, None when it came from the sitemap
        discovered: dict[str, str | None] = {}

        for index_url, collection in CATEGORY_INDEXES.items():
            try:
                html = await fetch_text(ctx.client, index_url, ctx.throttle, ctx.stats)
            except Exception as exc:
                logger.warning("uffizi: failed to fetch %s: %s", index_url, exc)
                continue

            for url in index_artwork_urls(html):
                if url not in ctx.seen:
                    discovered.setdefault(url, collection)

        # The current /artworks category pages are JS-rendered and may return
        # no links to fetch_text(). Use the sitemap as a complete URL index.
        if not discovered:
            try:
                sitemap_urls = await sitemap_artwork_urls(ctx)
            except Exception as exc:
                logger.warning("uffizi: sitemap discovery failed: %s", exc)
                sitemap_urls = []

            for url in sitemap_urls:
                if url not in ctx.seen:
                    discovered.setdefault(url, None)

            logger.info(
                "uffizi: sitemap fallback discovered %d artwork URLs",
                len(discovered),
            )

        if ctx.with_images:
            async for artwork in enrich_pool(
                ctx, list(discovered), fetch_one, ctx.limit
            ):
                yield artwork
        else:
            # Without fetching detail pages there is no reliable way to know
            # the collection for sitemap URLs. Only use directly discovered
            # category URLs in this mode.
            yielded = 0
            for url, collection in discovered.items():
                if collection is None:
                    continue
                yield Artwork(source="uffizi", url=url, image_url=None)
                yielded += 1
                if ctx.limit and yielded >= ctx.limit:
                    return
