"""The embedding model interface, and what every model's `encode_images` shares."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from PIL import Image

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class NoImagesError(ValueError):
    """None of the images given to `encode_images` could be read."""


def concat(embeddings: list[torch.Tensor], image_paths: list[str]) -> torch.Tensor:
    """The batches of `encode_images` as one tensor."""
    import torch

    if not embeddings:
        raise NoImagesError(f"none of the {len(image_paths)} images could be read")
    return torch.cat(embeddings)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for the model (e.g., 'clip', 'dinov2')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Full model identifier (e.g., 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def supports_text(self) -> bool:
        """Whether the model supports text-to-image similarity search."""
        ...

    @abstractmethod
    def load(self, device: str) -> None:
        """Load the model and processor onto the specified device."""
        ...

    @abstractmethod
    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, list[str]]:
        """Generate normalized embeddings for a list of images.

        Args:
            image_paths: List of paths to image files.
            batch_size: Batch size for processing.

        Returns:
            Tuple of:
            - Tensor of shape (n_images, embedding_dim) with L2-normalized embeddings.
            - List of successfully processed image paths (may be fewer than input if some failed).

        Raises:
            NoImagesError: None of the images could be read.
        """
        ...

    def encode_text(self, text: str) -> torch.Tensor | None:
        """Generate embedding for a text query.

        Args:
            text: The text query to encode.

        Returns:
            Tensor of shape (1, embedding_dim) with L2-normalized embedding,
            or None if the model doesn't support text encoding.
        """
        _ = text
        return None

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        """Generate embedding for a single PIL image.

        Args:
            image: A PIL Image object.

        Returns:
            Tensor of shape (1, embedding_dim) with L2-normalized embedding.
        """
        _ = image
        raise NotImplementedError("Subclass must implement encode_pil_image")

    def encode_pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Generate embeddings for images already in memory.

        This is the entry point for images that were streamed in rather than read
        from disk. The default encodes them one by one; models whose processor
        takes a list override it to use the GPU properly.
        """
        import torch

        return torch.cat([self.encode_pil_image(image) for image in images])


class _Forward(Protocol):
    def __call__(self, inputs: dict) -> torch.Tensor: ...


def encode_dataloader(
    dataloader, device: str | None, forward: _Forward
) -> tuple[list[torch.Tensor], list[str]]:
    """Run `forward` over every batch of a dataloader, normalising and collecting.

    `forward` takes the batch's tensors (everything but `path`), moved to `device`,
    and returns the raw (not yet normalised) embeddings.
    """
    import torch

    all_embeddings: list[torch.Tensor] = []
    processed_paths: list[str] = []
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i + 1}/{len(dataloader)}")
        if batch is None:
            logger.warning(f"Batch {i + 1} had no valid images, skipping")
            continue

        inputs = {k: v.to(device) for k, v in batch.items() if k != "path"}
        with torch.no_grad():
            embeddings = forward(inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.append(embeddings.cpu())
        processed_paths.extend(batch["path"])

    return all_embeddings, processed_paths


def unwrap_clip_output(output):
    """The embedding tensor out of a CLIP-style model output, whatever shape it took."""
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    elif hasattr(output, "text_embeds"):
        return output.text_embeds
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    else:
        return output
