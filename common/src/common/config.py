"""Typed project configuration: declared schema, then `glashelder.toml`, then the environment.

The schema below *is* the documentation and the defaults. Read it as:

    from common.config import config

    config().paths.embeddings.reference
    config().artmatcher.threshold

Attribute access rather than string keys, so a typo is a type error instead of a
silently missing value. Any setting can be overridden from the environment with a
`GLASHELDER_` prefix and `__` between levels:

    GLASHELDER_PATHS__IMAGES__GLASHELDER=/mnt/other/images
"""

import logging
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin

from . import PROJECT_ROOT

CONFIG_NAME = "glashelder.toml"
ENV_PREFIX = "GLASHELDER_"
ENV_SEPARATOR = "__"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImagePaths:
    glashelder: Path | None = None
    """Directory holding the glass plate images."""
    reference: Path | None = None
    """Directory holding the reference artwork images, if they are local at all."""
    cache: Path = Path(".image-cache")
    """Where fetched images land, only used when image caching is enabled."""


@dataclass(slots=True)
class EmbeddingPaths:
    glashelder: Path | None = None
    reference: Path | None = None
    """Embeddings of the reference artworks to match against."""


@dataclass(slots=True)
class Paths:
    images: ImagePaths = field(default_factory=ImagePaths)
    embeddings: EmbeddingPaths = field(default_factory=EmbeddingPaths)
    artworks: Path = Path("artworks.jsonl")
    """Artscraper output, which doubles as the artwork metadata index."""
    ml_model: Path | None = None


@dataclass(slots=True)
class Images:
    concurrency: int = 16
    """Image fetches in flight. Unrelated to the artscraper politeness rate."""
    cache: bool = False
    """Keep fetched images on disk. Off: a full reference set is tens of gigabytes."""


@dataclass(slots=True)
class Artmatcher:
    workers: int = 8
    threshold: float = 0.5
    top_k: int = 10


@dataclass(slots=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    images: Images = field(default_factory=Images)
    artmatcher: Artmatcher = field(default_factory=Artmatcher)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Build the config from `glashelder.toml` and the environment.

        `path` defaults to the nearest `glashelder.toml` at or above the working
        directory, falling back to the one next to the project root.
        """
        found = path if path is not None else find_config()
        data: dict[str, Any] = {}
        if found is not None and found.is_file():
            data = tomllib.loads(found.read_text())
            logger.debug("loaded config from %s", found)
        merge(data, env_overrides())
        return build(cls, data, prefix="")


def find_config() -> Path | None:
    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    candidate = PROJECT_ROOT / CONFIG_NAME
    return candidate if candidate.is_file() else None


def env_overrides() -> dict[str, Any]:
    """Nested dict of the `GLASHELDER_*` variables, `__` marking each level."""
    overrides: dict[str, Any] = {}
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX) or name == ENV_PREFIX:
            continue
        *sections, leaf = name.removeprefix(ENV_PREFIX).lower().split(ENV_SEPARATOR)
        target = overrides
        for section in sections:
            nested = target.setdefault(section, {})
            if not isinstance(nested, dict):
                nested = target[section] = {}
            target = nested
        target[leaf] = value
    return overrides


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Deep-merge `overlay` into `base`, in place."""
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merge(current, value)
        else:
            base[key] = value


def build[T](schema: type[T], data: dict[str, Any], prefix: str) -> T:
    """Instantiate a dataclass from a nested dict, coercing to the declared types."""
    known = {f.name: f for f in fields(cast(Any, schema))}
    if unknown := set(data) - set(known):
        where = f"{prefix.rstrip('.')} " if prefix else ""
        raise ValueError(
            f"unknown config {where}key(s): {', '.join(sorted(unknown))}. "
            f"known: {', '.join(sorted(known))}"
        )

    values = {}
    for name, spec in known.items():
        if name not in data:
            continue
        if is_dataclass(spec.type) and isinstance(spec.type, type):
            section = data[name]
            if not isinstance(section, dict):
                raise ValueError(f"config {prefix}{name} should be a table")
            values[name] = build(spec.type, section, prefix=f"{prefix}{name}.")
        else:
            values[name] = coerce(data[name], spec.type, f"{prefix}{name}")

    return schema(**values)


def coerce(value: Any, annotation: Any, where: str) -> Any:
    """Turn a TOML or environment value into the annotated type."""
    target = annotation
    if get_origin(annotation) in (Union, UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if value is None or value == "":
            return None
        if len(options) != 1:
            return value  # nothing sensible to pick, hand it over as-is
        target = options[0]

    if target is Path:
        return Path(value).expanduser()
    if target is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if target in (int, float, str):
        return target(value)
    return value


@cache
def config() -> Config:
    """The loaded config, built once per process."""
    return Config.load()


def defaults_for(section: object) -> dict[str, Any]:
    """A config section as a dict, for click's `ctx.default_map`.

    Only set values are handed over, so an unset option keeps the default declared
    on the command itself.
    """
    return {
        f.name: value
        for f in fields(cast(Any, section))
        if (value := getattr(section, f.name)) is not None and not is_dataclass(value)
    }
