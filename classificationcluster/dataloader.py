from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
        inputs["path"] = str(image_path)
        return inputs



def get_dataloader(image_paths: list[str], processor, num_workers: int = 8, batch_size: int = 64):
    dataset = ImageDataset(image_paths, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU->GPU transfer
        prefetch_factor=2,  # Batches to prefetch per worker
    )