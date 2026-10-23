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
    artworks: Path = Path("artworks.json.zst")
    """Artscraper output, which doubles as the artwork metadata index."""
    clusters: Path | None = None
    """Clustering result of the glashelder embeddings, see `cluster cluster`."""
    ml_model: Path | None = None
    """Set this when you want to train a model with `glashelder artmatcher
    train-classifier DATASET`. It saves the trained model to this path, and the
    labeling interface uses that model the next time it runs."""


@dataclass(slots=True)
class HostLimits:
    """Fetch limits for one host, overriding the `[images]` ones where set."""

    per_host: int | None = None
    rate: float | None = None


@dataclass(slots=True)
class Images:
    concurrency: int = 16
    """Image fetches in flight, over all hosts together."""
    per_host: int = 4
    """Image fetches in flight to any one host."""
    rate: float = 5.0
    """Requests per second to any one host, 0 for no limit."""
    hosts: dict[str, HostLimits] = field(default_factory=dict)
    """Limits for specific hosts, e.g. `[images.hosts."uploads.wikiart.org"]`."""
    cache: bool = False
    """Keep fetched images on disk. Off: a full reference set is tens of gigabytes."""

    def limits(
        self, host: str, per_host: int | None = None, rate: float | None = None
    ) -> tuple[int, float]:
        """`(per_host, rate)` for a host: its own table first, then the given defaults."""
        own = self.hosts.get(host, HostLimits())
        return (
            next(v for v in (own.per_host, per_host, self.per_host) if v is not None),
            next(v for v in (own.rate, rate, self.rate) if v is not None),
        )


@dataclass(slots=True)
class Embed:
    model: str = "clip"
    """Embedding model for `embed artworks` and `embed images`."""
    batch_size: int = 64


@dataclass(slots=True)
class Cluster:
    """Parameters of `cluster cluster`: UMAP, then HDBSCAN on its output."""

    rerun: str | None = None
    """Rerun viewer address to log the clustering to, e.g. `rerun+http://host:9876/proxy`."""
    n_neighbors: int = 30
    """UMAP `n_neighbors`: higher keeps more of the global structure."""
    min_dist: float = 0.0
    """UMAP `min_dist`: 0 packs similar images tightly, which HDBSCAN likes."""
    dimensions: int = 30
    """Dimensions UMAP reduces to before HDBSCAN (the 2D and 3D views come from these)."""
    min_cluster_size: int = 30
    """HDBSCAN `min_cluster_size`: the smallest group of images that counts as a cluster."""
    min_samples: int | None = None
    """HDBSCAN `min_samples`: higher makes more images noise. Unset means `min_cluster_size`."""
    soft: bool = True
    """Also save HDBSCAN's soft clustering, so `cluster serve` can move noise into clusters."""
    soft_top: int = 3
    """How many of its strongest clusters to save per image, with their scores."""


@dataclass(slots=True)
class Artmatcher:
    workers: int = 8
    threshold: float = 0.5
    top_k: int = 10
    candidates: int = 1
    """How many of the top CLIP matches the ML classifier tries while the ones
    before them are no match; the most confident is kept. Each extra one costs
    another ORB/hash extraction for every image without a match."""


@dataclass(slots=True)
class Glaswacht:
    results: Path = Path("combined_risk_assessment_results.csv")
    """CSV `glaswacht run` writes and `glaswacht serve` reads."""
    images: Path | None = None
    """Folder of assessed images `glaswacht serve` shows."""


@dataclass(slots=True)
class Ocr:
    base_url: str = "http://localhost:8000"
    """vLLM service base URL."""
    model: str | None = None
    """Model id; unset prompts for one interactively when the command needs it."""
    concurrency: int = 1
    """Parallel in-flight requests to vLLM."""
    cache: Path | None = None
    """Transcription cache dir. Unset keeps `$XDG_CACHE_HOME/glashelder/transcriptions`."""


@dataclass(slots=True)
class Artscraper:
    louvre_export: Path | None = None
    """Louvre export CSV, or a directory of them."""


@dataclass(slots=True)
class Config:
    secret: str | None = None
    """Secret for the serve commands' authentication. Unset means a random one
    per run. Better set from the environment (`GLASHELDER_SECRET`) than in the file."""
    paths: Paths = field(default_factory=Paths)
    images: Images = field(default_factory=Images)
    embed: Embed = field(default_factory=Embed)
    cluster: Cluster = field(default_factory=Cluster)
    artmatcher: Artmatcher = field(default_factory=Artmatcher)
    glaswacht: Glaswacht = field(default_factory=Glaswacht)
    ocr: Ocr = field(default_factory=Ocr)
    artscraper: Artscraper = field(default_factory=Artscraper)

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
            values[name] = build(
                spec.type, table(data, name, prefix), f"{prefix}{name}."
            )
        elif get_origin(spec.type) is dict and is_dataclass(get_args(spec.type)[1]):
            # a table of tables keyed by name, like `[images.hosts."some.host"]`
            item, where = get_args(spec.type)[1], f"{prefix}{name}."
            section = table(data, name, prefix)
            values[name] = {
                key: build(item, table(section, key, where), f"{where}{key}.")
                for key in section
            }
        else:
            values[name] = coerce(data[name], spec.type, f"{prefix}{name}")

    return schema(**values)


def table(data: dict[str, Any], name: str, prefix: str) -> dict[str, Any]:
    section = data[name]
    if not isinstance(section, dict):
        raise ValueError(f"config {prefix}{name} should be a table")
    return section


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

    Only set, plain values are handed over, so an unset option keeps the default
    declared on the command itself.
    """
    return {
        f.name: value
        for f in fields(cast(Any, section))
        if (value := getattr(section, f.name)) is not None
        and not is_dataclass(value)
        and not isinstance(value, dict)
    }
