"""Vlaamse Kunstcollectie Elasticsearch backend.

Retrieves artwork records directly from the Vlaamse Kunstcollectie Elasticsearch
index instead of scraping the public collection overview and individual pages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote

from ..model import Artwork
from . import Backend, Context

API_KEY = "***REMOVED***"
logger = logging.getLogger(__name__)
PIT_URL = "https://glue-vkc.es.europe-west1.gcp.cloud.es.io/production_vkc_records/_pit"
CLOSE_PIT_URL = "https://glue-vkc.es.europe-west1.gcp.cloud.es.io/_pit"

def _auth_headers() -> dict[str, str]:
    return {"authorization": f"ApiKey {API_KEY}", "content-type": "application/json"}

async def _open_pit(ctx: Context) -> str:
    resp = await ctx.client.post(PIT_URL, params={"keep_alive": "2m"}, headers=_auth_headers(), timeout=30.0)
    resp.raise_for_status()
    return resp.json()["id"]

async def _close_pit(ctx: Context, pit_id: str) -> None:
    try:
        await ctx.client.request(
            "DELETE", CLOSE_PIT_URL,
            content=json.dumps({"id": pit_id}),
            headers=_auth_headers(), timeout=30.0,
        )
    except Exception:
        logger.debug("kc: failed to close PIT", exc_info=True)

def _pit_search_body(pit_id: str, size: int, search_after: list | None) -> str:
    body: dict[str, Any] = {
        "size": size,
        "query": {"match_all": {}},
        "_source": [
            "slug", "title_nl", "description_nl", "image", "identifier",
            "dating", "location_nl", "subjects", "creators",
            "title_en", "title_fr", "title_de",
        ],
        "pit": {"id": pit_id, "keep_alive": "2m"},
        "sort": [{"_shard_doc": "asc"}],
        "track_total_hits": False,
    }
    if search_after is not None:
        body["search_after"] = search_after
    return json.dumps(body)


SEARCH_URL = "https://glue-vkc.es.europe-west1.gcp.cloud.es.io/_msearch"
INDEX = "production_vkc_records"
PAGE_SIZE = 24

# Afbeeldingen met dit pad worden niet opgenomen.
EXCLUDED_IMAGE_FRAGMENTS = ["public/11885", "public@27254"]

def _msearch_body(offset: int, size: int) -> str:
    """Build the newline-delimited Elasticsearch _msearch request body."""
    header = {"index": INDEX}

    query = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "filter": [],
                        "must": {"match_all": {}},
                    }
                },
                "random_score": {
                    "seed": 1790151604,
                    "field": "_seq_no",
                },
                "boost_mode": "replace",
            }
        },
        "size": size,
        "from": offset,
        "_source": [
            "slug",
            "title_nl",
            "description_nl",
            "image",
            "identifier",
            "dating",
            "location_nl",
            "subjects",
            "creators",
            "title_en",
            "title_fr",
            "title_de",
        ],
        "track_total_hits": True,
    }

    return json.dumps(header) + "\n" + json.dumps(query) + "\n"


def _text(value: Any) -> str | None:
    """Convert a scalar metadata value to readable text."""
    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    return None


def _label(value: Any) -> str | None:
    """Extract text from common Elasticsearch/JSON-LD label shapes."""
    direct = _text(value)
    if direct:
        return direct

    if isinstance(value, dict):
        for key in (
            "preferred_name",
            "preferred_name_nl",
            "name",
            "label",
            "value",
        ):
            result = _label(value.get(key))
            if result:
                return result

        return None

    if isinstance(value, list):
        values = [_label(item) for item in value]
        values = [item for item in values if item]

        return ", ".join(dict.fromkeys(values)) if values else None

    return None


def _image_url(value: Any) -> str | None:
    """Extract an allowed image URL from common VKC image field shapes."""
    direct = _text(value)

    if direct and direct.startswith(("http://", "https://")):
        decoded_url = unquote(direct)

        # Excludeer zowel gewone als URL-gecodeerde public/11885-paden.
        if any(fragment in decoded_url for fragment in EXCLUDED_IMAGE_FRAGMENTS):
            logger.debug(
                "kc: skipping excluded image: %s",
                direct,
            )
            return None

        if "/iiif/2/" in direct and "/full/" not in direct:
            return f"{direct}/full/1000,/0/default.jpg"

        return direct

    if isinstance(value, dict):
        for key in ("url", "src", "uri", "image_url", "href"):
            result = _image_url(value.get(key))
            if result:
                return result

        return None

    if isinstance(value, list):
        for item in value:
            result = _image_url(item)
            if result:
                return result

    return None


def _creator(record: dict[str, Any]) -> str | None:
    return _label(record.get("creators"))


def _year(record: dict[str, Any]) -> str | None:
    dating = _label(record.get("dating"))
    if dating:
        return dating


def _title(record: dict[str, Any]) -> str | None:
    for key in ("title_nl", "title_en", "title_fr", "title_de"):
        title = _label(record.get(key))
        if title:
            return title

    return None


def _record_url(record: dict[str, Any]) -> str | None:
    slug = _text(record.get("slug"))

    if slug and slug.startswith(("http://", "https://")):
        return slug

    if slug:
        return f"https://vlaamsekunstcollectie.be/collectie/{slug}"

    identifier = _text(record.get("identifier"))

    if identifier and identifier.startswith(("http://", "https://")):
        return identifier

    return None


def _to_artwork(record: dict[str, Any]) -> Artwork | None:
    """Convert one Elasticsearch hit source into an Artwork."""
    image_url = _image_url(record.get("image"))

    if not image_url:
        logger.debug(
            "kc: skipping record without allowed image: %r",
            record.get("identifier"),
        )
        return None

    url = _record_url(record)

    if not url:
        logger.debug(
            "kc: skipping record without URL: %r",
            record.get("identifier"),
        )
        return None

    return Artwork(
        source="kunstcollectie",
        url=url,
        image_url=image_url,
        title=_title(record),
        artist=_creator(record),
        year=_year(record),
    )

async def _search_page(
    ctx: Context,
    offset: int,
    size: int,
) -> tuple[list[Artwork], int]:
    """Fetch one Elasticsearch page.

    Returns:
        A tuple containing:
        - converted, allowed artworks;
        - raw Elasticsearch hit count, before excluded images are filtered out.
    """
    body = _msearch_body(offset, size)

    headers = {
        "content-type": "application/x-ndjson",
    }

    api_key = "***REMOVED***"

    if api_key:
        headers["authorization"] = f"ApiKey {api_key}"
    else:
        logger.warning(
            "kc: no Elasticsearch credentials configured; request may return 401"
        )

    response = await ctx.client.post(
        SEARCH_URL,
        content=body,
        headers=headers,
        timeout=30.0,
    )

    if response.status_code == 401:
        logger.error(
            "kc: Elasticsearch authentication failed: %s",
            response.text[:1000],
        )

    response.raise_for_status()

    payload = response.json()
    responses = payload.get("responses")

    if not isinstance(responses, list) or not responses:
        logger.warning("kc: Elasticsearch response has no responses array")
        return [], 0

    result = responses[0]

    if not isinstance(result, dict):
        return [], 0

    if result.get("error"):
        logger.warning(
            "kc: Elasticsearch query error: %s",
            result["error"],
        )
        return [], 0

    hits = result.get("hits", {}).get("hits", [])

    if not isinstance(hits, list):
        return [], 0

    raw_hit_count = len(hits)
    artworks: list[Artwork] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue

        source = hit.get("_source")

        if not isinstance(source, dict):
            continue

        artwork = _to_artwork(source)

        if artwork is not None:
            artworks.append(artwork)

    return artworks, raw_hit_count

class Kunstcollectie(Backend):
    """Retrieve artworks from the Vlaamse Kunstcollectie Elasticsearch index."""

    name = "kunstcollectie"

    async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
        yielded = 0
        seen: set[str] = set()
        pit_id = await _open_pit(ctx)

        try:
            search_after: list | None = None

            while True:
                body = _pit_search_body(pit_id, PAGE_SIZE, search_after)
                resp = await ctx.client.post(
                    "https://glue-vkc.es.europe-west1.gcp.cloud.es.io/_search",
                    content=body, headers=_auth_headers(), timeout=30.0,
                )
                resp.raise_for_status()
                payload = resp.json()

                pit_id = payload.get("pit_id", pit_id)  # ES may hand back a refreshed pit id
                hits = payload.get("hits", {}).get("hits", [])

                if not hits:
                    return

                for hit in hits:
                    source = hit.get("_source")
                    if isinstance(source, dict):
                        artwork = _to_artwork(source)
                        if artwork and artwork.url not in seen and artwork.url not in ctx.seen:
                            seen.add(artwork.url)
                            yield artwork
                            yielded += 1
                            if ctx.limit and yielded >= ctx.limit:
                                return

                search_after = hits[-1]["sort"]
        finally:
            await _close_pit(ctx, pit_id)