"""Embedding model abstraction layer with support for multiple models.

This module provides a unified interface for generating image embeddings
using different models like CLIP, DINOv2, and Qwen3-VL.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from common.images import is_url

if TYPE_CHECKING:
    import torch
    from transformers import (
        AutoImageProcessor,
        CLIPModel,
        CLIPProcessor,
        Dinov2Model,
        Qwen3VLModel,
        Qwen3VLProcessor,
    )

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get the best available device for tensor operations."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Model identifiers
CLIP_MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
DINOV2_MODEL_ID = "facebook/dinov2-large"
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"


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
        import torch

        from common.dataloader import get_dataloader

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        dataloader = get_dataloader(image_paths, self._processor, batch_size=batch_size)
        all_embeddings = []
        processed_paths = []

        for i, batch in enumerate(dataloader):
            logger.info(f"Batch {i + 1}/{len(dataloader)}")
            if batch is None:
                logger.warning(f"Batch {i + 1} had no valid images, skipping")
                continue

            inputs = {k: v.to(self._device) for k, v in batch.items() if k != "path"}

            with torch.no_grad():
                output = self._model.get_image_features(**inputs)
                # Handle case where output might be a model output object instead of tensor
                if hasattr(output, "image_embeds"):
                    embeddings = output.image_embeds
                elif hasattr(output, "pooler_output"):
                    embeddings = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    embeddings = output.last_hidden_state[:, 0]
                else:
                    embeddings = output
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
            processed_paths.extend(batch["path"])

        return torch.cat(all_embeddings), processed_paths

    def encode_text(self, text: str) -> torch.Tensor | None:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
                self._device
            )
            output = self._model.get_text_features(**inputs)
            # Handle case where output might be a model output object instead of tensor
            if hasattr(output, "text_embeds"):
                text_embedding = output.text_embeds
            elif hasattr(output, "pooler_output"):
                text_embedding = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                text_embedding = output.last_hidden_state[:, 0]
            else:
                text_embedding = output
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True)

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
            output = self._model.get_image_features(**inputs)
            # Handle case where output might be a model output object instead of tensor
            if hasattr(output, "image_embeds"):
                embedding = output.image_embeds
            elif hasattr(output, "pooler_output"):
                embedding = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                embedding = output.last_hidden_state[:, 0]
            else:
                embedding = output
            return embedding / embedding.norm(dim=-1, keepdim=True)


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
        import torch

        from common.dataloader import get_dataloader

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        dataloader = get_dataloader(image_paths, self._processor, batch_size=batch_size)
        all_embeddings = []
        processed_paths = []

        for i, batch in enumerate(dataloader):
            logger.info(f"Batch {i + 1}/{len(dataloader)}")
            if batch is None:
                logger.warning(f"Batch {i + 1} had no valid images, skipping")
                continue

            inputs = {k: v.to(self._device) for k, v in batch.items() if k != "path"}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
            processed_paths.extend(batch["path"])

        return torch.cat(all_embeddings), processed_paths

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


class Qwen3VLEmbedder(EmbeddingModel):
    """Qwen3-VL-Embedding-8B for image and text embeddings.

    This model requires ~16GB VRAM and supports both image and text encoding
    for multimodal similarity search.
    """

    def __init__(self):
        self._model: Qwen3VLModel | None = None
        self._processor: Qwen3VLProcessor | None = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "qwen3vl"

    @property
    def model_id(self) -> str:
        return QWEN3VL_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 4096

    @property
    def supports_text(self) -> bool:
        return True

    def load(self, device: str) -> None:
        import torch
        from transformers import Qwen3VLModel, Qwen3VLProcessor

        logger.info(f"Loading Qwen3-VL-Embedding model: {self.model_id}")
        self._model = Qwen3VLModel.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self._processor = Qwen3VLProcessor.from_pretrained(
            self.model_id, use_safetensors=False
        )
        self._model.to(torch.device(device))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        self._model.eval()
        self._device = device

    def _last_token_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embedding from the last valid token (EOS) position."""
        import torch

        # Find the last non-padding position for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 8,
    ) -> tuple[torch.Tensor, list[str]]:
        import torch
        from qwen_vl_utils import process_vision_info

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        all_embeddings = []
        processed_paths = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(image_paths) + batch_size - 1) // batch_size
            logger.info(f"Batch {batch_num}/{total_batches}")

            # Build batched messages
            batch_texts = []
            batch_images = []
            valid_paths = []

            for image_path in batch_paths:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"file://{image_path}",
                                    "min_pixels": 4096,
                                    "max_pixels": 1003520,
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ]
                    text = self._processor.apply_chat_template(
                        messages,  # ty:ignore[invalid-argument-type]
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    image_inputs, *_ = process_vision_info(messages)

                    batch_texts.append(text)
                    if image_inputs is not None:
                        batch_images.extend(image_inputs)
                    valid_paths.append(image_path)
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")
                    continue

            if not batch_texts:
                continue

            # Single batched forward pass
            inputs = self._processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = self._last_token_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.float().cpu())
                processed_paths.extend(valid_paths)

        return torch.cat(all_embeddings), processed_paths

    def encode_text(self, text: str) -> torch.Tensor | None:
        import torch

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

        formatted_text = self._processor.apply_chat_template(
            messages,  # ty:ignore[invalid-argument-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[formatted_text],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use last token pooling for consistency
            embedding = self._last_token_pooling(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.float()

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        import torch
        from qwen_vl_utils import process_vision_info

        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 4096,
                        "max_pixels": 1003520,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,  # ty:ignore[invalid-argument-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, *_ = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = self._last_token_pooling(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.float()


class OpenCLIPEmbedder(EmbeddingModel):
    """OpenCLIP-based image and text embeddings.

    Uses the open_clip library, which supports a wider range of CLIP model
    architectures and pretrained weights than HuggingFace transformers.

    Args:
        model_name: OpenCLIP model architecture (e.g., 'ViT-B-16', 'ViT-L-14').
        pretrained: Pretrained weights name (e.g., 'datacomp_xl_s13b_b90k').
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "datacomp_xl_s13b_b90k",
    ):
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "openclip"

    @property
    def model_id(self) -> str:
        return f"{self._model_name}/{self._pretrained}"

    @property
    def embedding_dim(self) -> int:
        assert self._model is not None, "Model not loaded. Call load() first."
        import open_clip

        return open_clip.get_model_config(self._model_name)["embed_dim"]

    @property
    def supports_text(self) -> bool:
        return True

    @property
    def device(self) -> str:
        assert self._device is not None, "Model not loaded. Call load() first."
        return self._device

    def load(self, device: str) -> None:
        import open_clip

        logger.info(f"Loading OpenCLIP model: {self.model_id}")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained, device=device
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model.eval()
        self._device = device

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, list[str]]:
        import torch

        assert self._model is not None and self._preprocess is not None

        all_embeddings = []
        processed_paths = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            logger.info(
                f"Batch {i // batch_size + 1}"
                f"/{(len(image_paths) + batch_size - 1) // batch_size}"
            )

            images = []
            valid_paths = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self._preprocess(img))
                    valid_paths.append(path)
                except Exception:
                    logger.warning(f"Failed to load image: {path}")

            if not images:
                continue

            batch_tensor = torch.stack(images).to(self._device)
            with torch.no_grad():
                embeddings = self._model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
            processed_paths.extend(valid_paths)

        return torch.cat(all_embeddings), processed_paths

    def encode_text(self, text: str) -> torch.Tensor | None:
        import torch

        assert self._model is not None and self._tokenizer is not None

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            text_embedding = self._model.encode_text(tokens)
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    def encode_pil_image(self, image: Image.Image) -> torch.Tensor:
        import torch

        assert self._model is not None and self._preprocess is not None

        img = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._model.encode_image(img)
            return embedding / embedding.norm(dim=-1, keepdim=True)


# Registry of available embedding models
EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {
    "clip": CLIPEmbedder,
    "openclip": OpenCLIPEmbedder,
    "dinov2": DINOv2Embedder,
    "qwen3vl": Qwen3VLEmbedder,
}


def get_embedder(model_name: str = "clip") -> EmbeddingModel:
    """Get an embedding model instance by name.

    Args:
        model_name: Name of the model ('clip', 'dinov2', or 'qwen3vl').

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


def _common_root(image_paths: list[str]) -> str | None:
    """Infer the directory all image paths live under, or None if not derivable."""
    import os.path

    absolute = [p for p in image_paths if p and os.path.isabs(p) and not is_url(p)]
    if not absolute:
        return None
    if len(absolute) == 1:
        return os.path.dirname(absolute[0])
    return os.path.commonpath(absolute)


def relativize_image_paths(
    image_paths: list[str], root: str | Path | None
) -> tuple[list[str], str | None]:
    """Strip `root` from image paths so they can be rebased on another machine.

    Falls back to the common ancestor of the given paths when `root` is not
    provided, which is what makes legacy pickles (absolute paths, no stored
    root) portable as well.

    Returns:
        Tuple of (relative paths, the root they are relative to).
    """
    root_str = str(root) if root is not None else _common_root(image_paths)
    if root_str is None:
        return list(image_paths), None

    root_path = Path(root_str)
    relative = []
    for p in image_paths:
        # "" is a placeholder for a missing image; Path("") would become "."
        # a URL is already portable and must not be mangled into a path
        if p == "" or is_url(p):
            relative.append(p)
            continue
        candidate = Path(p)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(root_path)
            except ValueError:
                logger.warning(f"Image path {p} is not under root {root_str}")
        relative.append(str(candidate))

    return relative, root_str


def rebase_paths(image_paths: list[str], root: str | Path | None) -> list[str]:
    """Re-root image paths recorded on another machine onto `root`.

    This is the single place where "paths were stored somewhere else" is fixed:
    the stored paths are made relative to their common ancestor and re-joined
    with the local image directory. Empty entries (used as placeholders) and
    already-relative paths are preserved. Returns the paths unchanged when
    `root` is None.
    """
    if root is None:
        return list(image_paths)

    root_path = Path(root)
    relative, stored_root = relativize_image_paths(image_paths, None)
    if stored_root is not None and str(root_path) != stored_root:
        logger.info(f"Rebasing image paths from {stored_root} onto {root_path}")

    return [str(root_path / p) if p and not is_url(p) else p for p in relative]


def read_embeddings(path: str | Path, root: str | Path | None = None) -> dict:
    """What an embeddings file holds, a store or a legacy pickle, as stored.

    Local image paths come back relative to `data["root"]`: the root recorded in
    the file, or `root`, or for old pickles without one, the directory all their
    absolute paths share. This is the raw form, for converting between formats;
    `load_embeddings` is what gives paths usable on this machine.

    Returns:
        Dictionary containing 'images', 'embeddings', 'root', 'model',
        'model_id' and 'embedding_dim'. For a store the embeddings are a
        memory-mapped tensor, nothing is read until it is used.
    """
    from .store import Store, is_store

    if is_store(path):
        import torch

        store = Store.open(path)
        return {
            "model": store.model,
            "model_id": store.model_id,
            "embedding_dim": store.dim,
            "root": store.root,
            "images": store.refs(),
            "embeddings": torch.from_numpy(store.vectors()),
        }

    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle legacy format without model metadata
    if "model" not in data:
        logger.warning(
            "Loading embeddings in legacy format (no model metadata). "
            "Assuming CLIP embeddings."
        )
        data["model"] = "clip"
        data["model_id"] = CLIP_MODEL_ID
        data["embedding_dim"] = 768

    stored_root = root if root is not None else data.get("root") or None
    data["images"], data["root"] = relativize_image_paths(data["images"], stored_root)
    return data


def load_embeddings(path: str | Path, root: str | Path | None = None) -> dict:
    """Load embeddings from a store or a legacy pickle.

    Image paths are always returned under `root`, which is the single place
    where the mismatch between the machine that generated the embeddings and
    the machine reading them gets resolved: the file keeps paths relative to
    the root it was made with, and here they are joined onto this one.

    Args:
        path: Path to the store directory or the pickle file.
        root: Directory the images live in on *this* machine. When omitted, the
            root recorded in the file (or inferred from it) is used, which
            only works if the images are in the same place as when embedded.

    Returns:
        Dictionary containing 'images' (absolute paths under `root`, or URLs),
        'embeddings', 'root', and model metadata.
    """
    data = read_embeddings(path)

    resolved_root = root if root is not None else data["root"]
    if resolved_root is None:
        if any(p and not is_url(p) for p in data["images"]):
            logger.warning(
                f"Embeddings at {path} have no image root and none was given; "
                "using image paths as stored."
            )
        return data

    data["root"] = str(resolved_root)
    data["images"] = rebase_paths(data["images"], resolved_root)

    return data


def save_embeddings(
    path: str,
    image_paths: list[str],
    embeddings: torch.Tensor,
    embedder: EmbeddingModel,
    root: str | Path | None = None,
) -> None:
    """Save embeddings to a pickle file with model metadata.

    Image paths are stored relative to `root` (inferred from the paths when not
    given) so that the pickle stays usable when the image directory moves.

    Args:
        path: Path to save the pickle file.
        image_paths: List of image paths.
        embeddings: Tensor of embeddings.
        embedder: The embedding model used to generate the embeddings.
        root: Directory the image paths are under.
    """
    import pickle

    relative_paths, stored_root = relativize_image_paths(image_paths, root)

    data = {
        "model": embedder.name,
        "model_id": embedder.model_id,
        "embedding_dim": embedder.embedding_dim,
        "root": stored_root,
        "images": relative_paths,
        "embeddings": embeddings,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Saved embeddings to {path}")
    logger.info(f"  Model: {embedder.name} ({embedder.model_id})")
    logger.info(f"  Image root: {stored_root}")
    logger.info(f"  Images: {len(image_paths)}")
    logger.info(f"  Embedding dim: {embedder.embedding_dim}")
