from bs4 import BeautifulSoup

from artscraper.backends.parse import artwork_json_ld, first_text, parse_json_ld, text


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_text() -> None:
    assert text("  Autumn ") == "Autumn"
    assert text("   ") is None
    assert text(1573) is None
    assert text(None) is None


def test_first_text() -> None:
    assert first_text([None, " ", 3, " b ", "c"]) == "b"
    assert first_text([]) is None


def test_json_ld_flattens_graphs_and_lists() -> None:
    page = soup(
        """
        <script type="application/ld+json">{"@graph": [{"@type": "WebPage"}, {"@type": "Person"}]}</script>
        <script type="application/ld+json">not json</script>
        <script type="application/ld+json"></script>
        <script type="application/ld+json">[{"@type": "VisualArtwork", "name": "x"}, 3]</script>
        """
    )
    assert [item["@type"] for item in parse_json_ld(page)] == [
        "WebPage",
        "Person",
        "VisualArtwork",
    ]
    assert artwork_json_ld(page) == {"@type": "VisualArtwork", "name": "x"}


def test_artwork_json_ld_fallbacks() -> None:
    # without an artwork type, something with a name and an image will do
    page = soup(
        '<script type="application/ld+json">[{"@type": ["Thing"]}, {"name": "x", "image": "y"}]</script>'
    )
    assert artwork_json_ld(page) == {"name": "x", "image": "y"}

    # and failing that, the first object
    page = soup('<script type="application/ld+json">{"@type": "WebPage"}</script>')
    assert artwork_json_ld(page) == {"@type": "WebPage"}

    assert artwork_json_ld(soup("<html></html>")) is None
