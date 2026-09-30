"""Shared loading logic for wikiart/glashelder embeddings and classifiers."""

import glob
import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import torch

import embed
from artscraper import Artwork
from embed import get_device
from embed.search import place

from .ml_classifier import MLClassifier

logger = logging.getLogger(__name__)


@dataclass
class MatchingInputs:
    device: str
    model: str  # the reconciled model name actually used
    wikiart_emb: torch.Tensor
    wikiart_paths: list[str]
    glashelder_emb: torch.Tensor
    glashelder_paths: list[str]


def load_matching_inputs(
    *,
    wikiart_embeddings: str,
    glashelder_embeddings: str | None,
    model: str,
    image_dir: str | None,
    wikiart_root: str | Path | None = None,
    glashelder_root: str | Path | None = None,
    batch_size: int = 64,
    artworks: dict[str, Artwork] | None = None,
    sources: Collection[str] = (),
) -> MatchingInputs:
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load wikiart embeddings
    logger.info(f"Loading wikiart embeddings from {wikiart_embeddings}...")
    wikiart_data = embed.load_embeddings(wikiart_embeddings, root=wikiart_root)
    wikiart_emb, wikiart_paths = wikiart_data["embeddings"], wikiart_data["images"]
    if sources:
        wikiart_emb, wikiart_paths = only_sources(
            wikiart_emb, wikiart_paths, artworks or {}, sources
        )
    wikiart_emb = place(wikiart_emb, device, label=f"reference {wikiart_embeddings}")
    embedding_model_name = wikiart_data.get("model", "clip")
    logger.info(f"Loaded {len(wikiart_paths)} wikiart images")
    logger.info(f"Embedding model: {embedding_model_name}")

    if embedding_model_name != model:
        logger.warning(
            f"Wikiart embeddings use model '{embedding_model_name}' but --model is '{model}'. "
            f"Using '{embedding_model_name}' for consistency."
        )
        model = embedding_model_name

    # Load or generate glashelder embeddings
    if glashelder_embeddings is not None:
        logger.info(f"Loading glashelder embeddings from {glashelder_embeddings}...")
        glashelder_data = embed.load_embeddings(
            glashelder_embeddings, root=glashelder_root
        )
        glashelder_emb = glashelder_data["embeddings"].to(device)
        glashelder_paths: list[str] = glashelder_data["images"]
        logger.info(f"Loaded {len(glashelder_paths)} glashelder images")
    else:
        logger.info(f"Loading embedder model: {model}")
        embedder = embed.get_embedder(model)
        embedder.load(device)

        glashelder_paths = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))
        logger.info(f"Found {len(glashelder_paths)} glashelder images")

        if not glashelder_paths:
            raise ValueError(f"No images found in {image_dir}")

        logger.info("Generating glashelder embeddings (this may take a while)...")
        glashelder_emb, glashelder_paths = embedder.encode_images(
            glashelder_paths, batch_size=batch_size
        )
        glashelder_emb = glashelder_emb.to(device)
        logger.info(f"Generated embeddings for {len(glashelder_paths)} images")

    return MatchingInputs(
        device=device,
        model=model,
        wikiart_emb=wikiart_emb,
        wikiart_paths=wikiart_paths,
        glashelder_emb=glashelder_emb,
        glashelder_paths=glashelder_paths,
    )


def only_sources(
    embeddings: torch.Tensor,
    refs: list[str],
    artworks: dict[str, Artwork],
    sources: Collection[str],
) -> tuple[torch.Tensor, list[str]]:
    """The references whose artwork was scraped from one of `sources`.

    This copies the selected rows out of the store into memory, which is what
    lets a smaller selection also fit in less RAM or VRAM.
    """
    if not artworks:
        raise ValueError("--source needs the artworks (--artworks) to know the sources")
    known = {artwork.source for artwork in artworks.values()}
    if unknown := set(sources) - known:
        raise ValueError(
            f"unknown source(s): {', '.join(sorted(unknown))}. "
            f"known: {', '.join(sorted(known))}"
        )

    keep = [
        i
        for i, ref in enumerate(refs)
        if (artwork := artworks.get(ref)) is not None and artwork.source in sources
    ]
    logger.info(
        f"{len(keep):,} of {len(refs):,} references are from {', '.join(sorted(sources))}"
    )
    return embeddings[keep], [refs[i] for i in keep]


def load_classifier(ml_model: str | None, threshold: float) -> "MLClassifier | None":
    if not ml_model:
        return None
    logger.info(f"Loading ML classifier from {ml_model}...")
    classifier = MLClassifier.load(ml_model, threshold=threshold)
    logger.info("ML classifier loaded")
    return classifier
