"""Checks for the parsers, the only parts that silently rot when a site changes.

Run with `uv run python artscraper/tests/test_parsers.py`.
"""

from bs4 import BeautifulSoup

from artscraper.backends.cle import parse_metadata as parse_cleveland
from artscraper.backends.kikirpa import full_image_url
from artscraper.backends.kikirpa import parse_page as parse_kikirpa
from artscraper.backends.uffizi import artwork_urls
from artscraper.backends.uffizi import parse_page as parse_uffizi
from artscraper.backends.wikiart import parse_artwork_url, parse_image_url

WIKIART_PAGE = """
<meta property="og:image" content="https://uploads1.wikiart.org/autumn-1573(1).jpg!Large.jpg" />
<meta property="og:title" content="Autumn - Giuseppe Arcimboldo - WikiArt.org" />
"""

KIKIRPA_PAGE = """
<script type="application/ld+json">
{"@type": "VisualArtwork", "name": "tenaille", "identifier": "20066235",
 "url": "https://balat.kikirpa.be/en/object/20066235/",
 "image": "https://iiif.kikirpa.be/iiif/2/B001427/full/!800,800/0/default.jpg",
 "creator": {"@type": "Person", "name": "inconnu"}, "dateCreated": "1401/1500"}
</script>
"""


def test_wikiart_url() -> None:
    artwork = parse_artwork_url(
        "https://www.wikiart.org/en/giuseppe-arcimboldo/autumn-1573"
    )
    assert artwork is not None
    assert (artwork.artist, artwork.title, artwork.year) == (
        "Giuseppe Arcimboldo",
        "Autumn",
        "1573",
    )

    # a title that only looks like it ends in a year keeps it
    numbered = parse_artwork_url("https://www.wikiart.org/en/artist/opus-9999")
    assert (
        numbered is not None and numbered.year is None and numbered.title == "Opus 9999"
    )

    # listing pages are not artworks
    assert parse_artwork_url("https://www.wikiart.org/en/popular-paintings") is None


def test_wikiart_image() -> None:
    assert (
        parse_image_url(BeautifulSoup(WIKIART_PAGE, "html.parser"))
        == "https://uploads1.wikiart.org/autumn-1573(1).jpg"
    )
    assert parse_image_url(BeautifulSoup("<html></html>", "html.parser")) is None


def test_kikirpa_page() -> None:
    artwork = parse_kikirpa(
        "https://balat.kikirpa.be/en/object/20066235/", KIKIRPA_PAGE
    )
    assert artwork is not None
    assert (artwork.title, artwork.artist, artwork.year) == (
        "tenaille",
        "inconnu",
        "1401/1500",
    )
    assert artwork.image_url is not None and "!1000,1000" in artwork.image_url

    assert (
        parse_kikirpa("https://balat.kikirpa.be/en/object/1/", "<html></html>") is None
    )


UFFIZI_PAGE = """
<script type="application/ld+json">
{"@graph": [{"@type": "WebPage", "name": "page"},
 {"@type": "VisualArtwork", "name": "Primavera",
  "creator": [{"name": "Sandro Botticelli "}, {"name": ""}]}]}
</script>
<h1>Primavera</h1>
<img srcset="https://www.datocms-assets.com/a.jpg 400w, https://www.datocms-assets.com/b.jpg 800w, https://www.datocms-assets.com/c.jpg 1600w">
<dl><dt>Date</dt><dd>1480</dd></dl>
"""

CLEVELAND_METADATA = """
{"metadata": {"title": " ", "name": "Twilight", "creator": ["A ", "", "B"], "date": 1890},
 "files": [{"name": "twilight_thumb.jpg"}, {"name": "twilight.jpg"},
           {"name": "twilight_original.jpg"}, {"name": "twilight.xml"}]}
"""


def test_uffizi_page() -> None:
    url = "https://www.uffizi.it/en/artworks/primavera"
    artwork = parse_uffizi(url, UFFIZI_PAGE)
    assert artwork is not None
    assert (artwork.url, artwork.title, artwork.artist, artwork.year) == (
        url,
        "Primavera",
        "Sandro Botticelli",
        "1480",
    )
    assert artwork.image_url == "https://www.datocms-assets.com/b.jpg"


def test_uffizi_urls() -> None:
    assert artwork_urls(
        [
            "/en/artworks/primavera/",
            "https://uffizi.it/en/artworks/primavera",
            "/en/artworks/search",
            "https://example.com/en/artworks/primavera",
            " /en/artworks/venus ",
        ]
    ) == [
        "https://www.uffizi.it/en/artworks/primavera",
        "https://www.uffizi.it/en/artworks/venus",
    ]


def test_cleveland_metadata() -> None:
    artwork = parse_cleveland("twilight-1", CLEVELAND_METADATA)
    assert artwork is not None
    assert (artwork.title, artwork.artist, artwork.year) == ("Twilight", "A; B", "1890")
    assert artwork.image_url == (
        "https://archive.org/download/twilight-1/twilight_original.jpg"
    )
    assert parse_cleveland("x", '{"metadata": {}, "files": []}') is None


def test_iiif_rewrite() -> None:
    thumbnail = "https://iiif.kikirpa.be/iiif/2/B001427/full/!280,280/0/default.jpg"
    assert full_image_url(thumbnail).endswith("/full/!1000,1000/0/default.jpg")
    # already full-size urls have no size to swap and must survive untouched
    maxed = "https://iiif.kikirpa.be/iiif/2/B001427/full/max/0/default.jpg"
    assert full_image_url(maxed) == maxed


if __name__ == "__main__":
    test_wikiart_url()
    test_wikiart_image()
    test_kikirpa_page()
    test_uffizi_page()
    test_uffizi_urls()
    test_cleveland_metadata()
    test_iiif_rewrite()
    print("ok")
