from .base import EmbeddingModel, NoImagesError
from .clip import CLIP_MODEL_ID, CLIPEmbedder
from .dinov2 import DINOv2Embedder

EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {
    "clip": CLIPEmbedder,
    "dinov2": DINOv2Embedder,
}


def get_embedder(model_name: str = "clip") -> EmbeddingModel:
    """Get an embedding model instance by name.

    Args:
        model_name: Name of the model ('clip' or 'dinov2').

    Returns:
        An EmbeddingModel instance (not yet loaded).

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name not in EMBEDDING_MODELS:
        available = ", ".join(EMBEDDING_MODELS.keys())
        raise ValueError(
            f"Unknown embedding model: {model_name}. Available: {available}"
        )
    return EMBEDDING_MODELS[model_name]()


__all__ = [
    "CLIP_MODEL_ID",
    "EMBEDDING_MODELS",
    "CLIPEmbedder",
    "DINOv2Embedder",
    "EmbeddingModel",
    "NoImagesError",
    "get_embedder",
]
