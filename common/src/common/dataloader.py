from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(self.image_paths[index]).convert("RGB")

        # Process returns dict with pixel_values, etc.
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["path"] = str(image_path)
        return inputs


class ImagePathDataset(Dataset[dict[str, Any]]):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(self.image_paths[index]).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["path"] = str(image_path)
        return inputs


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
    )
