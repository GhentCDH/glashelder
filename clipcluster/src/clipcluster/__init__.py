"""Clipcluster package - Image clustering and matching using CLIP embeddings."""

from clipcluster.cli import perform_search
from clipcluster.embed import (
    CLIPEmbedder,
    DINOv2Embedder,
    EmbeddingModel,
    Qwen3VLEmbedder,
    get_embedder,
    load_embeddings,
    save_embeddings,
)

__all__ = [
    "CLIPEmbedder",
    "DINOv2Embedder",
    "EmbeddingModel",
    "Qwen3VLEmbedder",
    "get_embedder",
    "load_embeddings",
    "perform_search",
    "save_embeddings",
]
