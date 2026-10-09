"""Wereldmuseum collection backend (collectie.wereldmuseum.nl).

The site is an Angular SPA backed by a CollectionConnect (cc) ASMX web
service.  There are no sitemaps; records are discovered via the search
API at /cc/ccConnector.asmx/search.

Only records **without** a "work in copyright" watermark are included.
Copyright items are detected by checking whether the image proxy URL
points to ``CopyRightImage.jpg`` — the museum serves that placeholder
for every record whose image rights are not cleared.

The following thesaurus categories (under *Object name* >
*beeldmateriaal* > *beeldmateriaal naar medium of techniek*) are scraped:

  - prenten (prints)
  - schilderingen (paintings)
  - pastels
  - tekeningen (drawings)
  - beeldhouwwerken naar type onderwerp (sculptures by subject type)
  - schetsen (beeldende werken) (sketches)
  - bebouwde omgeving (built environment)
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

from bs4 import BeautifulSoup

from ..client import fetch_json
from ..model import Artwork
from . import Backend, Context

BASE_URL = "https://collectie.wereldmuseum.nl"
SEARCH_URL = f"{BASE_URL}/cc/ccConnector.asmx/search"
HANDLE_BASE = "https://hdl.handle.net/20.500.11840"


# Thesaurus concept numbers (CN) for the requested categories.
# These live under Object name (OVM.AAG) > beeldmateriaal (OVM.AAG.AAC) >
# beeldmateriaal naar medium of techniek (OVM.AAG.AAC.AAE), except for
# "bebouwde omgeving" which is a direct child of Object name.
CATEGORIES: dict[str, str] = {
    "bebouwde omgeving": "OVM.AAG.AAB",
    "prenten": "OVM.AAG.AAC.AAE.AAF",
    "schilderingen": "OVM.AAG.AAC.AAE.AAG",
    "pastels": "OVM.AAG.AAC.AAE.AAJ",
    "tekeningen": "OVM.AAG.AAC.AAE.AAL",
    "beeldhouwwerken naar type onderwerp": "OVM.AAG.AAC.AAE.AAN.AAG",
    "schetsen (beeldende werken)": "OVM.AAG.AAC.AAE.AAQ",
}


PAGE_SIZE = 60
COPYRIGHT_MARKER = "CopyRightImage.jpg"
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  API helpers
# --------------------------------------------------------------------------- #


def _search_spec(cn: str, label: str, first: int) -> dict:
    """Build a searchSpec for a thesaurus facet value.

    The searchSpec is the JSON object that the CollectionConnect ``search``
    endpoint expects.  Setting ``facetValues`` with a ``CN|label`` query
    value restricts results to one thesaurus term.
    """
    return {
        "language": "nl",
        "fallbackLanguage": "en",
        "ccSettingsName": "Alternative",
        "numPerPage": PAGE_SIZE,
        "first": first,
        "showtype": "icons",
        "searchscope": "Search the whole record",
        "advancedSearchscope": "Search the whole record",
        "customSearchscope": "Search the whole record",
        "searchValues": [
            {
                "searchscope": {
                    "scopeName": "Search the whole record",
                    "scopeTag": "",
                },
                "id": 0,
                "tag": "All fields",
                "value": "",
                "searchOperator": "and",
            }
        ],
        "searchValuesAdvanced": [],
        "searchValuesCustom": [],
        "searchValuesProfessional": [],
        "facetValues": {"Object name": [f"{cn}|{label}"]},
        "basketFilters": [],
        "queryFilters": [],
        "filters": [],
        "sortfields": {
            "sortfield": [
                {
                    "sortfield": "Object number",
                    "field": "/record/sortnumber",
                    "asc": True,
                }
            ]
        },
    }


async def _post_json(ctx: Context, url: str, payload: dict) -> dict:
    return await fetch_json(
        ctx.client, ctx.throttle, ctx.stats, "POST", url, json=payload
    )


# --------------------------------------------------------------------------- #
#  HTML parsing
# --------------------------------------------------------------------------- #


# Extract a trailing year / century from the "origin + date" field.
# Examples: "Japan 1966-1969", "China 12de eeuw", "Jerusalem 1948-1965"
_YEAR_RE = re.compile(
    r"(\d{1,4}(?:ste|de)?\s*eeuw"  # "12de eeuw", "19ste eeuw"
    r"|\d{4}(?:\s*[-–]\s*\d{4})?)",  # "1966-1969", "2018" # noqa:RUF001
    re.IGNORECASE,
)


def _full_image_url(src: str) -> str | None:
    """Rebuild the image-proxy URL at a usable resolution.

    The museum proxy expects ``server=localhost&port=17581`` (those are
    internal references the proxy resolves, not the user's machine).
    We keep them, swap the size, and drop the passepartout / border
    decorations.
    """
    parsed = urlparse(src)
    qs = parse_qs(parsed.query)
    filename = qs.get("filename", [""])[0]
    if not filename:
        return None

    params = {
        "server": "localhost",
        "port": "17581",
        "filename": filename,
        "width": str(IMAGE_WIDTH),
        "height": str(IMAGE_HEIGHT),
        "borderwidth": "0",
        "borderheight": "0",
        "bordercolor": "e8e8e8",
        "bg": "ffffff",
        "cache": "yes",
    }
    # safe="/" keeps the slashes in the filename path unencoded.
    return urljoin(BASE_URL, f"{parsed.path}?{urlencode(params, safe='/')}")


def _extract_year(origin_date: str) -> str | None:
    """Pull a year / century string from the origin+date field.

    Returns ``None`` when no date or century pattern is found, rather
    than falling back to the full origin string.
    """
    if not origin_date:
        return None
    m = _YEAR_RE.search(origin_date)
    return m.group(1).strip() if m else None


def _parse_results(html: str) -> list[dict]:
    """Extract records from the search-result HTML table.

    Returns **all** cells (including copyright ones) so the caller can
    distinguish "page had records but all were copyright" from "page was
    empty / end of results".  Each dict carries an ``is_copyright`` flag.

    The HTML returned by the ASMX endpoint is malformed (mismatched
    tags, stray ``</div>``), so we anchor on ``div.ccIconviewRecordImage``
    and look for the sibling data div rather than relying on a clean
    ``<td>`` structure.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []

    for img_div in soup.select("div.ccIconviewRecordImage"):
        img_tag = img_div.find("img")
        if not img_tag:
            continue

        # Find the metadata div — either a sibling or within the same <td>.
        data_div = img_div.find_next_sibling("div", class_="ccIconviewRecordData")
        if data_div is None:
            parent_td = img_div.find_parent("td")
            if parent_td is not None:
                data_div = parent_td.find("div", class_="ccIconviewRecordData")
        if data_div is None:
            continue

        src = img_tag.get("src", "")
        dragging_image = img_tag.get("data-draggingimage", "")
        record_id = img_tag.get("data-draggingid", "")

        # Copyright placeholder appears in both src and data-draggingimage.
        is_copyright = (src is not None and COPYRIGHT_MARKER in src) or (
            dragging_image is not None and COPYRIGHT_MARKER in dragging_image
        )

        divs = [d.get_text(strip=True) for d in data_div.find_all("div")]
        # Layout: [0]=category, [1]=title, [2]=artist, [3]=origin+date,
        #         [4]=object number, [5..8]=misc (usually empty)
        title = divs[1] if len(divs) > 1 else ""
        artist = divs[2] if len(divs) > 2 else ""
        origin_date = divs[3] if len(divs) > 3 else ""

        out.append(
            {
                "is_copyright": is_copyright,
                "record_id": record_id,
                "title": title or None,
                "artist": artist or None,
                "year": _extract_year(origin_date),
                "image_src": src,
            }
        )

    return out


# --------------------------------------------------------------------------- #
#  Backend
# --------------------------------------------------------------------------- #


class Wereldmuseum(Backend):
    name = "wereldmuseum"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        yielded = 0

        for label, cn in CATEGORIES.items():
            if ctx.limit and yielded >= ctx.limit:
                return

            logger.info("wereldmuseum: category %s (%s)", label, cn)
            first = 1

            while True:
                spec = _search_spec(cn, label, first)
                try:
                    resp = await _post_json(
                        ctx,
                        SEARCH_URL,
                        {"authToken": "", "searchSpec": spec},
                    )
                except Exception as exc:
                    logger.error("wereldmuseum: search failed for %s: %s", label, exc)
                    break

                data = resp.get("d", {})
                if not data:
                    break

                total = data.get("resultCount", 0)
                html = data.get("result", "")

                # "Geen zoekresultaten" = no results at all
                if not html or "Geen zoekresultaten" in html:
                    logger.info("wereldmuseum: no results for %s", label)
                    break

                records = _parse_results(html)

                # Distinguish "page had cells" from "page was empty".
                # Even if every cell is copyright we must keep paginating,
                # because later pages may contain usable images.
                if not records:
                    break

                page_yielded = 0
                for rec in records:
                    if rec["is_copyright"]:
                        continue
                    rid = rec["record_id"]
                    if not rid:
                        continue
                    url = f"{HANDLE_BASE}/{rid}"
                    if url in ctx.seen:
                        continue
                    ctx.seen.add(url)

                    image_url = _full_image_url(rec["image_src"])
                    if not image_url:
                        continue

                    yield Artwork(
                        source="wereldmuseum",
                        url=url,
                        image_url=image_url,
                        title=rec["title"],
                        artist=rec["artist"],
                        year=rec["year"],
                    )
                    yielded += 1
                    page_yielded += 1

                    if ctx.limit and yielded >= ctx.limit:
                        logger.info(
                            "wereldmuseum: %s — %d cells, %d yielded "
                            "(page %d, total %d) — limit reached",
                            label,
                            len(records),
                            page_yielded,
                            (first - 1) // PAGE_SIZE + 1,
                            total,
                        )
                        return

                logger.info(
                    "wereldmuseum: %s — %d cells, %d yielded (page %d, total %d)",
                    label,
                    len(records),
                    page_yielded,
                    (first - 1) // PAGE_SIZE + 1,
                    total,
                )

                first += PAGE_SIZE
                if first > total:
                    break
