"""Checks for the layered config.

Run with `uv run python common/tests/test_config.py`.
"""

import os
import tempfile
from pathlib import Path

from common.config import Config, defaults_for

TOML = """
[paths]
artworks = "some/artworks.jsonl"

[paths.images]
glashelder = "~/plates"

[images]
concurrency = 4
cache = true

[artmatcher]
threshold = 0.9
"""


def write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def test_defaults() -> None:
    config = Config()
    assert config.paths.artworks == Path("artworks.json.zst")
    assert config.images.concurrency == 16
    assert config.paths.images.glashelder is None


def test_toml_reaches_the_right_fields() -> None:
    config = Config.load(write(TOML))
    assert config.paths.artworks == Path("some/artworks.jsonl")
    assert config.paths.images.glashelder == Path.home() / "plates"
    assert config.images.concurrency == 4 and config.images.cache is True
    assert config.artmatcher.threshold == 0.9
    # untouched keys keep their declared defaults
    assert config.artmatcher.workers == 8


def test_env_beats_the_file() -> None:
    path = write(TOML)
    os.environ["GLASHELDER_PATHS__ARTWORKS"] = "/tmp/from-env.jsonl"
    os.environ["GLASHELDER_IMAGES__CONCURRENCY"] = "32"
    os.environ["GLASHELDER_IMAGES__CACHE"] = "false"
    try:
        config = Config.load(path)
    finally:
        for name in (
            "GLASHELDER_PATHS__ARTWORKS",
            "GLASHELDER_IMAGES__CONCURRENCY",
            "GLASHELDER_IMAGES__CACHE",
        ):
            del os.environ[name]

    assert config.paths.artworks == Path("/tmp/from-env.jsonl")
    # strings from the environment come back as the declared type
    assert config.images.concurrency == 32 and isinstance(
        config.images.concurrency, int
    )
    assert config.images.cache is False


def test_host_limits() -> None:
    config = Config.load(
        write(
            TOML
            + '\n[images.hosts."uploads.wikiart.org"]\nrate = 20\n'
            + '\n[images.hosts."slow.example"]\nper_host = 1\n'
        )
    )
    images = config.images
    assert images.hosts["uploads.wikiart.org"].rate == 20.0
    # a host's own table wins, then the given defaults, then `[images]`
    assert images.limits("uploads.wikiart.org") == (4, 20.0)
    assert images.limits("uploads.wikiart.org", per_host=2, rate=1.0) == (2, 20.0)
    assert images.limits("slow.example", rate=1.0) == (1, 1.0)
    assert images.limits("other.host") == (4, 5.0)

    try:
        Config.load(write('[images.hosts."a.b"]\nrat = 1\n'))
    except ValueError as e:
        assert "rat" in str(e) and "images.hosts.a.b" in str(e)
    else:
        raise AssertionError("a misspelled host key should not be ignored")


def test_unknown_key_raises() -> None:
    try:
        Config.load(write("[artmatcher]\nthreshhold = 0.5\n"))
    except ValueError as e:
        assert "threshhold" in str(e) and "artmatcher" in str(e)
    else:
        raise AssertionError("a misspelled key should not be ignored")


def test_click_defaults() -> None:
    config = Config.load(write(TOML))
    assert defaults_for(config.artmatcher) == {
        "workers": 8,
        "threshold": 0.9,
        "top_k": 10,
        "candidates": 1,
    }
    # nested sections and unset values are left out
    assert "images" not in defaults_for(config.paths)
    assert "ml_model" not in defaults_for(config.paths)


if __name__ == "__main__":
    test_defaults()
    test_toml_reaches_the_right_fields()
    test_env_beats_the_file()
    test_host_limits()
    test_unknown_key_raises()
    test_click_defaults()
    print("ok")
