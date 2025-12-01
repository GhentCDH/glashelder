from ultralytics import YOLO
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb limit
import os
from tqdm import tqdm
from itertools import chain, islice
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from queue import Queue
import threading
import logging


logger = logging.getLogger(__name__)

class YOLOCropper:
    def __init__(self, model_path, device="gpu"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def _load_if_needed(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def crop(self, pil_image, imgsz=1056):
        self._load_if_needed()

        img_np = np.array(pil_image)

        results = self.model.predict(
            source=img_np,
            imgsz=imgsz,
            save=False,
            verbose=False,
            device=self.device,
        )[0]

        if len(results.boxes) == 0:
            return pil_image

        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].tolist())
        return pil_image.crop((x1, y1, x2, y2))



def run_crop_on_batch_of_images(model_path, image_folder, output_folder, device="cpu", batch_size=20, num_workers=4, prefetch=10):


    cropper = YOLOCropper(model_path, device=device)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = list(
        chain.from_iterable(image_folder.glob(ext) for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff"))
    )
    
    def load_image(image_path):
        """Load and convert image to RGB."""
        try:
            img = Image.open(image_path).convert("RGB")
            return img, image_path.name
        except Exception as e:
            logger.error(f"Error loading {image_path.name}: {e}")
            return None, None
    
    def save_batch(batch_items):
        """Save a batch of images to disk in parallel."""
        def save_single(item):
            cropped, name = item
            try:
                cropped.save(output_folder / name, optimize=False)
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(save_single, batch_items))
    
    # Use ThreadPoolExecutor to parallelize image loading
    batch = []
    with ThreadPoolExecutor(max_workers=num_workers) as loader_executor:
        # Submit initial batch of image loading tasks
        future_to_path = {loader_executor.submit(load_image, path): path 
                          for path in image_paths[:prefetch]}
        remaining_paths = iter(image_paths[prefetch:])
        
        with tqdm(total=len(image_paths), desc="Processing images", unit="image") as pbar:
            while future_to_path:
                for future in as_completed(future_to_path):
                    img, name = future.result()
                    del future_to_path[future]
                    
                    if img is not None:
                        cropped_img = cropper.crop(img)
                        batch.append((cropped_img.copy(), name))
                        img.close()
                    
                    pbar.update(1)
                    try:
                        next_path = next(remaining_paths)
                        future_to_path[loader_executor.submit(load_image, next_path)] = next_path
                    except StopIteration:
                        pass
                    if len(batch) >= batch_size:
                        save_batch(batch)
                        batch.clear()
    
    if batch:
        save_batch(batch)