"""MegaLoc place descriptors (vision-only, no text support).

MegaLoc (Berton and Masone, CVPR 2025 workshops) is trained to give photos of
the same place similar vectors, also from another viewpoint, which a general
model like DINOv2 is not. A DINOv2 ViT-B backbone with SALAD aggregation gives
8448 values per image. The model code is vendored in `megaloc_model.py`; the
weights come from Hugging Face at a fixed revision, so neither can change under
a store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from .base import EmbeddingModel, concat, encode_dataloader

if TYPE_CHECKING:
    import torch

    from .megaloc_model import MegaLoc

MEGALOC_MODEL_ID = "gberton/MegaLoc"
REVISION = "a0f34722c4297ff787e022433799250180860af7"
SIZE = 322
"""Input side in pixels, as in the paper's evaluation (a multiple of 14)."""
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)


class Processor:
    """MegaLoc's evaluation preprocessing, called like a Hugging Face processor:
    RGB, resized to SIZE x SIZE, ImageNet normalized."""

    def __init__(self):
        import torch
        from torchvision.transforms import v2

        self.transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize((SIZE, SIZE), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(MEAN, STD),
            ]
        )

    def __call__(
        self, images: Image.Image | list[Image.Image], return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        import torch

        batch = images if isinstance(images, list) else [images]
        pixels = torch.stack([self.transform(image.convert("RGB")) for image in batch])
        return {"pixel_values": pixels}


class MegaLocEmbedder(EmbeddingModel):
    """MegaLoc place descriptors (vision-only, no text support)."""

    def __init__(self):
        self._model: MegaLoc | None = None
        self._processor: Processor | None = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "megaloc"

    @property
    def model_id(self) -> str:
        return MEGALOC_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 8448

    @property
    def supports_text(self) -> bool:
        return False

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from .megaloc_model import MegaLoc

        logger.info(f"Loading MegaLoc model: {self.model_id}@{REVISION[:7]}")
        weights = hf_hub_download(self.model_id, "model.safetensors", revision=REVISION)
        model = MegaLoc()
        model.load_state_dict(load_file(weights))
        self._model = model.to(torch.device(device)).eval()
        self._processor = Processor()
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
            dataloader, self._device, lambda inputs: model(inputs["pixel_values"])
        )
        return concat(embeddings, image_paths), processed_paths

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        return self.encode_pil_batch([image])

    def encode_pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        with torch.no_grad():
            pixels = self._processor(images)["pixel_values"].to(self._device)
            embedding = self._model(pixels)
            # already L2 normalized by the model; again, so float error cannot drift
            return embedding / embedding.norm(dim=-1, keepdim=True)
