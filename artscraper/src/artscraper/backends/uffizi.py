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

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..client import fetch_text
from ..model import Artwork
from . import Backend, Context, enrich_pool

BASE_URL = "https://www.uffizi.it"
INDEX_URLS = (
    f"{BASE_URL}/en/the-uffizi/prints-and-drawings/artworks",
    f"{BASE_URL}/en/the-uffizi/painting/artworks",
    f"{BASE_URL}/en/the-uffizi/sculpture/artworks",
)

ARTWORK_RE = re.compile(r"^/en/artworks/(?!search/?$)[^/?#]+/?$")
logger = logging.getLogger(__name__)


def artwork_urls(html: str) -> list[str]:
    """Extract Uffizi artwork detail URLs from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        absolute = urljoin(BASE_URL, href)
        parsed = urlparse(absolute)

        if parsed.netloc not in {"www.uffizi.it", "uffizi.it"}:
            continue
        if not ARTWORK_RE.fullmatch(parsed.path):
            continue

        url = f"{BASE_URL}{parsed.path.rstrip('/')}"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def _collection_from_page(html: str) -> str | None:
    """Read the artwork Collection value from the rendered detail page."""
    soup = BeautifulSoup(html, "html.parser")

    # The Uffizi detail pages render:
    #   Date -> ... -> Museum -> ... -> Collection -> Painting/Sculpture/...
    # Try the semantic HTML first.
    text = soup.get_text(" ", strip=True)
    match = re.search(
        r"\bCollection\s+(Painting|Sculpture|Department of Prints and Drawings)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    # JSON-LD sometimes contains the collection as category/keywords.
    for data in parse_json_ld(html):
        for key in ("about", "genre", "category", "keywords"):
            value = data.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, dict):
                    item = item.get("name")
                if not isinstance(item, str):
                    continue
                lowered = item.lower()
                if "prints and drawings" in lowered:
                    return "department of prints and drawings"
                if lowered == "painting":
                    return "painting"
                if lowered == "sculpture":
                    return "sculpture"

    return None


async def sitemap_artwork_urls(ctx: Context) -> list[str]:
    """Fallback discovery for the current Uffizi site.

    The three /artworks index pages are client-rendered and can contain no
    artwork anchors in the HTML returned to the scraper. The site's sitemap
    is therefore used as a complete URL index when that happens.
    """
    sitemap_url = f"{BASE_URL}/sitemap.xml"
    raw = await fetch_text(ctx.client, sitemap_url, ctx.throttle, ctx.stats)

    # Sitemap files are XML, but fetch_text gives us the response as text.
    try:
        from xml.etree import ElementTree

        root = ElementTree.fromstring(raw.lstrip("\ufeff").encode())
    except Exception:
        logger.warning("uffizi: could not parse %s", sitemap_url)
        return []

    urls: list[str] = []
    seen: set[str] = set()

    for loc in root.iter():
        if loc.tag.rsplit("}", 1)[-1] != "loc" or not loc.text:
            continue

        value = loc.text.strip()
        parsed = urlparse(value)
        if parsed.netloc not in {"www.uffizi.it", "uffizi.it"}:
            continue
        if not ARTWORK_RE.fullmatch(parsed.path):
            continue

        url = f"{BASE_URL}{parsed.path.rstrip('/')}"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def parse_json_ld(html: str) -> list[dict[str, Any]]:
    """Return schema.org JSON-LD objects found on an artwork page."""
    soup = BeautifulSoup(html, "html.parser")
    result: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            graph = value.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    add(item)
            else:
                result.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text()
        if not raw.strip():
            continue
        try:
            add(json.loads(raw))
        except json.JSONDecodeError:
            continue

    return result


def _first_json_ld(data: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the JSON-LD object that looks most like the artwork."""
    for item in data:
        item_type = item.get("@type")
        if item_type in {"VisualArtwork", "CreativeWork", "Artwork"}:
            return item
        if item.get("name") and (item.get("image") or item.get("creator")):
            return item
    return data[0] if data else None


def _person_name(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        return str(name).strip() if name else None
    if isinstance(value, list):
        names = [_person_name(item) for item in value]
        names = [name for name in names if name]
        return ", ".join(names) if names else None
    if isinstance(value, str):
        return value.strip() or None
    return None


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


def _image_from_html(html: str, data: dict[str, Any] | None) -> str | None:
    """Return a single middle-resolution artwork image URL."""
    soup = BeautifulSoup(html, "html.parser")

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


def _date_from_page(html: str) -> str | None:
    """Extract the artwork Date value from the Uffizi detail page."""
    soup = BeautifulSoup(html, "html.parser")

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


def parse_page(
    url: str,
    html: str,
    expected_collection: str | None = None,
) -> Artwork | None:
    data = _first_json_ld(parse_json_ld(html))
    soup = BeautifulSoup(html, "html.parser")

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

    image_url = _image_from_html(html, data)

    if image_url is None:
        logger.warning(
            "uffizi: no image found at %s; img=%d source=%d jsonld=%s",
            url,
            len(soup.find_all("img")),
            len(soup.find_all("source")),
            bool(data),
        )
        return None

    date = _date_from_page(html)

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

    return Artwork(
        source="uffizi",
        url=(data or {}).get("url") or url,
        image_url=image_url,
        title=title,
        artist=artist,
        year=(data or {}).get("dateCreated") or (data or {}).get("datePublished"),
    )


class Uffizi(Backend):
    name = "uffizi"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        # First try the supplied category index pages. On the current Uffizi
        # site these pages are client-rendered, so the HTML may contain zero
        # artwork links. In that case fall back to the site's sitemap.
        category_indexes = (
            (
                f"{BASE_URL}/en/the-uffizi/prints-and-drawings/artworks",
                "department of prints and drawings",
            ),
            (
                f"{BASE_URL}/en/the-uffizi/painting/artworks",
                "painting",
            ),
            (
                f"{BASE_URL}/en/the-uffizi/sculpture/artworks",
                "sculpture",
            ),
        )

        discovered: list[tuple[str, str | None]] = []
        seen: set[str] = set()

        for index_url, collection in category_indexes:
            try:
                html = await fetch_text(ctx.client, index_url, ctx.throttle, ctx.stats)
            except Exception as exc:
                logger.warning("uffizi: failed to fetch %s: %s", index_url, exc)
                continue

            urls = artwork_urls(html)
            for url in urls:
                if url not in ctx.seen and url not in seen:
                    seen.add(url)
                    discovered.append((url, collection))

        # The current /artworks category pages are JS-rendered and may return
        # no links to fetch_text(). Use the sitemap as a complete URL index.
        if not discovered:
            try:
                sitemap_urls = await sitemap_artwork_urls(ctx)
            except Exception as exc:
                logger.warning("uffizi: sitemap discovery failed: %s", exc)
                sitemap_urls = []

            for url in sitemap_urls:
                if url not in ctx.seen and url not in seen:
                    seen.add(url)
                    discovered.append((url, None))

            logger.info(
                "uffizi: sitemap fallback discovered %d artwork URLs",
                len(discovered),
            )

        async def fetch_one(
            ctx: Context, item: tuple[str, str | None]
        ) -> Artwork | None:
            url, _expected_collection = item

            html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)

            return parse_page(url, html, None)

        if ctx.with_images:
            # enrich_pool expects URLs, so wrap the tuple in a local fetcher.
            async def fetch_wrapped(ctx: Context, url: str) -> Artwork | None:
                for candidate, collection in discovered:
                    if candidate == url:
                        return await fetch_one(ctx, (candidate, collection))
                return None

            urls = [url for url, _ in discovered]
            async for artwork in enrich_pool(ctx, urls, fetch_wrapped, ctx.limit):
                yield artwork
        else:
            # Without fetching detail pages there is no reliable way to know
            # the collection for sitemap URLs. Only use directly discovered
            # category URLs in this mode.
            yielded = 0
            for url, collection in discovered:
                if collection is None:
                    continue
                yield Artwork(source="uffizi", url=url, image_url=None)
                yielded += 1
                if ctx.limit and yielded >= ctx.limit:
                    return
