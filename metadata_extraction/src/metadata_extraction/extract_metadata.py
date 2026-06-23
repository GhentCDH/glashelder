"""Pull candidate named-entity phrases out of OCR'd caption text and look
them up against Wikidata.

The OCR'd lines (museum captions, lantern-slide labels, engraved inscriptions)
are short, multilingual, and often ALL-CAPS, which makes statistical NER
models unreliable. Capitalization is a much stronger signal here than a
trained model: candidate phrases are just runs of capitalized/ALL-CAPS words,
bridged across lowercase connector words ("van", "de", "del", ...) so multi-
word names like "Andrea del Sarto" or "Université de Gand" survive intact.
Wikidata's own search does the real disambiguation/filtering.
"""

import difflib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

USER_AGENT = "ghentcdh-glashelder-metadata-extraction/0.1 (https://www.ghentcdh.ugent.be/). Contact: joren.six@ugent.be"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# lowercase words that may bridge two capitalized words into one phrase
# without breaking it (articles/prepositions across the languages in this
# corpus: French, Dutch, German, Italian, Latin, English)
_CONNECTORS = {
    "a",
    "al",
    "aus",
    "d",
    "da",
    "dai",
    "dal",
    "dalla",
    "de",
    "degli",
    "dei",
    "del",
    "della",
    "delle",
    "den",
    "der",
    "des",
    "di",
    "do",
    "dos",
    "du",
    "e",
    "el",
    "en",
    "et",
    "het",
    "l",
    "la",
    "le",
    "les",
    "mit",
    "of",
    "op",
    "qu",
    "s",
    "sur",
    "ten",
    "ter",
    "the",
    "und",
    "van",
    "von",
    "y",
}

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# clause/sentence punctuation that separates unrelated caption fields (e.g.
# "CHAUMONT. - Château. Entrée" is three fields, not one run-on phrase).
# Word-internal hyphens ("Südost-Kongo-Afrika") are deliberately not split.
_SEGMENT_RE = re.compile(r'[.,;:!?()\[\]{}"“”]|--+|[\u2013\u2014]|\s-\s')


def _is_capitalized(token: str) -> bool:
    return len(token) >= 2 and token[0].isupper() and not token.isupper()


def _is_allcaps(token: str) -> bool:
    return len(token) >= 3 and token.isupper()


def _is_connector(token: str) -> bool:
    return token.lower() in _CONNECTORS


def _phrases_in_segment(text: str) -> list[str]:
    phrases = []
    chunk: list[str] = []

    def flush():
        while chunk and _is_connector(chunk[-1]):
            chunk.pop()
        if chunk:
            phrases.append(" ".join(chunk))
        chunk.clear()

    for match in _TOKEN_RE.finditer(text):
        token = match.group()
        if _is_capitalized(token) or _is_allcaps(token):
            chunk.append(token)
        elif _is_connector(token) and chunk:
            chunk.append(token.lower())
        else:
            flush()
    flush()

    return phrases


def extract_candidate_phrases(text: str) -> list[str]:
    phrases = []
    for segment in _SEGMENT_RE.split(text):
        phrases.extend(_phrases_in_segment(segment))
    return phrases


def candidate_phrases_for_record(record: dict) -> list[str]:
    """Unique candidate phrases across a record's 'meta' and 'content' lines."""
    lines = [*record.get("meta", []), *record.get("content", [])]
    seen: dict[str, None] = {}
    for line in lines:
        for phrase in extract_candidate_phrases(line):
            seen.setdefault(phrase, None)
    return list(seen)


@dataclass
class WikidataMatch:
    id: str
    label: str
    description: str | None
    url: str
    match_type: str | None  # "label", "alias", or "description" - what was hit
    match_text: str | None  # the actual text that matched (may differ from label)
    similarity: float  # query-to-label string similarity, in [0, 1] - not a probability


def _similarity(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(), 3)


def search_wikidata(
    query: str, lang: str = "en", limit: int = 3, timeout: float = 10.0
) -> list[WikidataMatch]:
    """Query Wikidata's wbsearchentities API for a candidate phrase."""
    params = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "search": query,
            "language": lang,
            "format": "json",
            "limit": limit,
        }
    )
    req = urllib.request.Request(
        f"{WIKIDATA_API}?{params}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)

    matches = []
    for result in data.get("search", []):
        label = result.get("label", query)
        match = result.get("match", {})
        matches.append(
            WikidataMatch(
                id=result["id"],
                label=label,
                description=result.get("description"),
                url=result.get(
                    "concepturi", f"https://www.wikidata.org/wiki/{result['id']}"
                ),
                match_type=match.get("type"),
                match_text=match.get("text"),
                similarity=_similarity(query, label),
            )
        )
    return matches
