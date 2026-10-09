"""Checks for moving recorded image paths onto another root."""

from common.paths import _common_root, rebase_paths, relativize_image_paths

URL = "https://example.org/a.jpg"


def test_common_root():
    assert _common_root(["/data/x/a.jpg", "/data/y/b.jpg"]) == "/data"
    assert _common_root(["/data/x/a.jpg"]) == "/data/x"
    assert _common_root(["a.jpg", URL, ""]) is None


def test_relativize_keeps_placeholders_and_urls():
    relative, root = relativize_image_paths(["/data/a.jpg", "", URL], "/data")
    assert relative == ["a.jpg", "", URL]
    assert root == "/data"


def test_relativize_infers_the_root():
    relative, root = relativize_image_paths(["/data/x/a.jpg", "/data/y/b.jpg"], None)
    assert relative == ["x/a.jpg", "y/b.jpg"]
    assert root == "/data"


def test_relativize_without_root_is_unchanged():
    assert relativize_image_paths(["a.jpg", URL], None) == (["a.jpg", URL], None)


def test_rebase_onto_local_root():
    paths = ["/remote/set/x/a.jpg", "/remote/set/y/b.jpg", "", URL]
    assert rebase_paths(paths, "/local") == [
        "/local/x/a.jpg",
        "/local/y/b.jpg",
        "",
        URL,
    ]


def test_rebase_relative_paths():
    assert rebase_paths(["x/a.jpg"], "/local") == ["/local/x/a.jpg"]


def test_rebase_without_root_is_unchanged():
    paths = ["/remote/a.jpg"]
    assert rebase_paths(paths, None) == paths
