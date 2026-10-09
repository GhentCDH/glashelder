import pytest

from embed.cli import display_name, parse_top_k
from embed.embed import NoImagesError, concat


@pytest.mark.parametrize(
    ("ref", "name"),
    [
        ("/data/images/a.jpg", "a.jpg"),
        ("https://example.org/images/a%20b.jpg", "a b.jpg"),
        ("https://example.org/images/a.jpg?size=large", "a.jpg"),
        ("https://example.org/", "example.org"),
    ],
)
def test_display_name(ref, name):
    assert display_name(ref) == name


@pytest.mark.parametrize(
    ("value", "top_k"),
    [(None, 12), ("", 12), ("5", 5), ("0", None), ("-3", None), ("abc", None)],
)
def test_parse_top_k(value, top_k):
    assert parse_top_k(value) == top_k


def test_concat_without_batches():
    with pytest.raises(NoImagesError):
        concat([], ["a.jpg"])
