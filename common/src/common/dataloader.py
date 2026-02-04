import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 0.5  # seconds, doubles each retry


def _load_image_with_retry(image_path: str | Path) -> Image.Image | None:
    """Load an image with retry logic for transient errors (e.g., network shares)."""
    for attempt in range(MAX_RETRIES):
        try:
            return Image.open(image_path).convert("RGB")
        except PermissionError:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2**attempt)
                time.sleep(delay)
            else:
                logger.warning(
                    f"Failed to load image after {MAX_RETRIES} retries: {image_path}"
                )
                return None
        except OSError as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    return None


class ImageDataset(Dataset[dict[str, Any] | None]):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        image_path = self.image_paths[index]
        image = _load_image_with_retry(image_path)

        if image is None:
            return None

        # Process returns dict with pixel_values, etc.
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["path"] = str(image_path)
        return inputs


class ImagePathDataset(Dataset[dict[str, Any] | None]):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> dict[str, Any] | None:
        image_path = self.image_paths[index]
        image = _load_image_with_retry(image_path)

        if image is None:
            return None

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["path"] = str(image_path)
        return inputs


def _collate_skip_none(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Custom collate function that filters out None values from failed image loads."""
    import torch

    # Filter out None values
    filtered: list[dict[str, Any]] = [item for item in batch if item is not None]

    if not filtered:
        return None

    # Standard collate logic for dict of tensors
    result: dict[str, Any] = {}
    for key in filtered[0]:
        if key == "path":
            result[key] = [item[key] for item in filtered]
        else:
            result[key] = torch.stack([item[key] for item in filtered])

    return result


def get_dataloader(
    image_paths: list[str] | list[Path],
    processor,
    num_workers: int = 8,
    batch_size: int = 64,
    dataset_type: str = "image",
):
    datasets = {
        "image": ImageDataset,
        "imagepath": ImagePathDataset,
    }

    dataset = datasets[dataset_type](image_paths, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU->GPU transfer
        prefetch_factor=8,  # Batches to prefetch per worker
        collate_fn=_collate_skip_none,
    )
