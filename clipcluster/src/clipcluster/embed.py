"""Embedding model abstraction layer with support for multiple models.

This module provides a unified interface for generating image embeddings
using different models like CLIP, DINOv2, and Qwen3-VL.
"""

import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)

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
        return None


class CLIPEmbedder(EmbeddingModel):
    """CLIP-based image and text embeddings."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    @property
    def name(self) -> str:
        return "clip"

    @property
    def model_id(self) -> str:
        return CLIP_MODEL_ID

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def supports_text(self) -> bool:
        return True

    def load(self, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {self.model_id}")
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()
        self._device = device

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, list[str]]:
        from common.dataloader import get_dataloader

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
                embeddings = self._model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu())
            processed_paths.extend(batch["path"])

        return torch.cat(all_embeddings), processed_paths

    def encode_text(self, text: str) -> torch.Tensor | None:
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
                self._device
            )
            text_embedding = self._model.get_text_features(**inputs)
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True)


class DINOv2Embedder(EmbeddingModel):
    """DINOv2-based image embeddings (vision-only, no text support)."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

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
        from transformers import AutoImageProcessor, Dinov2Model

        logger.info(f"Loading DINOv2 model: {self.model_id}")
        self._model = Dinov2Model.from_pretrained(self.model_id)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> tuple[torch.Tensor, list[str]]:
        from common.dataloader import get_dataloader

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


class Qwen3VLEmbedder(EmbeddingModel):
    """Qwen3-VL-Embedding-8B for image and text embeddings.

    This model requires ~16GB VRAM and supports both image and text encoding
    for multimodal similarity search.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

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
        from transformers import Qwen3VLModel, Qwen3VLProcessor

        logger.info(f"Loading Qwen3-VL-Embedding model: {self.model_id}")
        self._model = Qwen3VLModel.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self._processor = Qwen3VLProcessor.from_pretrained(self.model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device

    def _last_token_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embedding from the last valid token (EOS) position."""
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
        from qwen_vl_utils import process_vision_info

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
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, _ = process_vision_info(messages)

                    batch_texts.append(text)
                    batch_images.extend(
                        image_inputs
                    )  # Usually returns a list with one image
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

        formatted_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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


# Registry of available embedding models
EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {
    "clip": CLIPEmbedder,
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


def load_embeddings(path: str) -> dict:
    """Load embeddings from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        Dictionary containing 'images', 'embeddings', and optionally model metadata.
    """
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

    return data


def save_embeddings(
    path: str,
    image_paths: list[str],
    embeddings: torch.Tensor,
    embedder: EmbeddingModel,
) -> None:
    """Save embeddings to a pickle file with model metadata.

    Args:
        path: Path to save the pickle file.
        image_paths: List of image paths.
        embeddings: Tensor of embeddings.
        embedder: The embedding model used to generate the embeddings.
    """
    import pickle

    data = {
        "model": embedder.name,
        "model_id": embedder.model_id,
        "embedding_dim": embedder.embedding_dim,
        "images": image_paths,
        "embeddings": embeddings,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Saved embeddings to {path}")
    logger.info(f"  Model: {embedder.name} ({embedder.model_id})")
    logger.info(f"  Images: {len(image_paths)}")
    logger.info(f"  Embedding dim: {embedder.embedding_dim}")
