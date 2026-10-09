"""DINOv2-based image embeddings (vision-only, no text support)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from .base import EmbeddingModel, concat, encode_dataloader

if TYPE_CHECKING:
    import torch
    from transformers import AutoImageProcessor, Dinov2Model

DINOV2_MODEL_ID = "facebook/dinov2-large"

logger = logging.getLogger(__name__)


class DINOv2Embedder(EmbeddingModel):
    """DINOv2-based image embeddings (vision-only, no text support)."""

    def __init__(self):
        self._model: Dinov2Model | None = None
        self._processor: AutoImageProcessor | None = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "dinov2"

    @property
    def model_id(self) -> str:
        return DINOV2_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def supports_text(self) -> bool:
        return False

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, Dinov2Model

        logger.info(f"Loading DINOv2 model: {self.model_id}")
        self._model = Dinov2Model.from_pretrained(self.model_id, use_safetensors=False)
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_safetensors=False
        )
        self._model.to(torch.device(device))  # ty:ignore[invalid-argument-type]
        self._model.eval()
        self._device = device

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, list[str]]:
        from embed.dataloader import get_dataloader

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        model = self._model
        dataloader = get_dataloader(image_paths, self._processor, batch_size=batch_size)
        embeddings, processed_paths = encode_dataloader(
            dataloader,
            self._device,
            lambda inputs: model(**inputs).last_hidden_state[:, 0],
        )
        return concat(embeddings, image_paths), processed_paths

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        return self._encode_pil(image)

    def encode_pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        # the DINOv2 processor takes a list, so a streamed batch is one forward pass
        return self._encode_pil(images)

    def _encode_pil(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        with torch.no_grad():
            processed = self._processor(images=images, return_tensors="pt")  # ty:ignore[call-non-callable]
            inputs = processed.to(self._device)
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]
            return embedding / embedding.norm(dim=-1, keepdim=True)
