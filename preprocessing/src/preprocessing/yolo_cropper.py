from ultralytics import YOLO
import numpy as np

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
            print("nothing")
            return pil_image

        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].tolist())
        return pil_image.crop((x1, y1, x2, y2))



def run_crop_on_batch_of_images(model_path, image_folder, output_folder, device="cpu"):
    from pathlib import Path
    from PIL import Image
    import os

    cropper = YOLOCropper(model_path, device=device)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff"):
        for image_path in image_folder.glob(ext):
            with Image.open(image_path).convert("RGB") as img:
                cropped_img = cropper.crop(img)
                cropped_img.save(output_folder / image_path.name)