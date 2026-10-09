"""CLIP-based image and text embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from .base import EmbeddingModel, concat, encode_dataloader, unwrap_clip_output

if TYPE_CHECKING:
    import torch
    from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

logger = logging.getLogger(__name__)


class CLIPEmbedder(EmbeddingModel):
    """CLIP-based image and text embeddings."""

    def __init__(self, model_id: str = CLIP_MODEL_ID):
        self._model_id = model_id
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "clip"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        assert self._model is not None, "Model not loaded. Call load() first."
        return self._model.config.projection_dim

    @property
    def supports_text(self) -> bool:
        return True

    @property
    def model(self) -> CLIPModel:
        """Get the underlying CLIP model. Raises if not loaded."""
        assert self._model is not None, "Model not loaded. Call load() first."
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        """Get the CLIP processor. Raises if not loaded."""
        assert self._processor is not None, "Processor not loaded. Call load() first."
        return self._processor

    @property
    def device(self) -> str:
        """Get the device the model is loaded on. Raises if not loaded."""
        assert self._device is not None, "Model not loaded. Call load() first."
        return self._device

    def load(self, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {self.model_id}")
        self._model = CLIPModel.from_pretrained(self.model_id, use_safetensors=False)
        self._processor = CLIPProcessor.from_pretrained(
            self.model_id, use_safetensors=False
        )
        self._model.to(device)  # ty: ignore[invalid-argument-type]
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
            lambda inputs: unwrap_clip_output(model.get_image_features(**inputs)),
        )
        return concat(embeddings, image_paths), processed_paths

    def encode_text(self, text: str) -> torch.Tensor | None:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
                self._device
            )
            output = unwrap_clip_output(self._model.get_text_features(**inputs))
            return output / output.norm(dim=-1, keepdim=True)

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        return self._encode_pil(image)

    def encode_pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        # the CLIP processor takes a list, so a streamed batch is one forward pass
        return self._encode_pil(images)

    def _encode_pil(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt").to(
                self._device
            )
            embedding = unwrap_clip_output(self._model.get_image_features(**inputs))
            return embedding / embedding.norm(dim=-1, keepdim=True)
